use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::cors::{Any, CorsLayer};

const SOLVER_SOCKET: &str = "/tmp/captop-solver.sock";

#[derive(Deserialize)]
struct SolveRequest {
    image: String,
}

#[derive(Serialize)]
struct SolveResponse {
    text: String,
}

#[derive(Deserialize)]
struct ReportRequest {
    image: String,
    prediction: String,
}

struct AppState {
    solver_socket: String,
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        solver_socket: SOLVER_SOCKET.to_string(),
    });

    let governor_config = Arc::new(
        GovernorConfigBuilder::default()
            .key_extractor(tower_governor::key_extractor::SmartIpKeyExtractor)
            .per_second(6)
            .burst_size(2)
            .finish()
            .unwrap(),
    );

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_private_network(true);

    let app = Router::new()
        .route("/solve", post(solve))
        .route("/report", post(report))
        .route("/health", axum::routing::get(health))
        .layer(GovernorLayer {
            config: governor_config,
        })
        .layer(cors)
        .with_state(state)
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse::<u16>()
        .unwrap();

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("API running at http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn solve(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SolveRequest>,
) -> Result<Json<SolveResponse>, StatusCode> {
    let image_data = STANDARD
        .decode(&payload.image)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Connect to the solver service
    let mut stream = UnixStream::connect(&state.solver_socket)
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    // Send length-prefixed image data
    let len = image_data.len() as u32;
    stream
        .write_all(&len.to_le_bytes())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    stream
        .write_all(&image_data)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    stream
        .shutdown()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut text = String::new();
    stream
        .read_to_string(&mut text)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if text.len() != 6 { save("malformed", &image_data, &text); }
    Ok(Json(SolveResponse { text }))
}

async fn report(Json(req): Json<ReportRequest>) -> StatusCode {
    let _ = STANDARD.decode(&req.image).map(|data| save("failed", &data, &req.prediction));
    StatusCode::OK
}

fn save(dir: &str, data: &[u8], pred: &str) {
    let ts = std::time::UNIX_EPOCH.elapsed().unwrap_or_default().as_secs();
    let _ = std::fs::create_dir_all(dir).map(|_| std::fs::write(format!("{dir}/{ts}_{pred}.jpg"), data));
} 

async fn health() -> StatusCode {
    StatusCode::OK
}
