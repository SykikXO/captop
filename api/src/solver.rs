use image::ImageReader;
use ndarray::Array4;
use ort::{GraphOptimizationLevel, Session};
use std::io::Cursor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

const CHARS: &str = "0123456789abcdefghijklmnopqrstuvwxyz";
const SOCKET_PATH: &str = "/tmp/captop-solver.sock";

#[tokio::main]
async fn main() {
    // Clean up stale socket
    let _ = std::fs::remove_file(SOCKET_PATH);

    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file("model/captcha_model.onnx")
        .unwrap();

    let session = std::sync::Arc::new(session);

    let listener = UnixListener::bind(SOCKET_PATH).unwrap();
    println!("Solver listening on {}", SOCKET_PATH);

    loop {
        let (stream, _) = listener.accept().await.unwrap();
        let session = session.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, &session).await {
                eprintln!("Solver error: {}", e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::UnixStream,
    session: &Session,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read 4-byte length prefix (little-endian u32)
    let len = stream.read_u32_le().await?;

    // Read image bytes
    let mut image_data = vec![0u8; len as usize];
    stream.read_exact(&mut image_data).await?;

    // Process
    let tensor = preprocess(&image_data)?;

    let outputs = session.run(ort::inputs!["image" => tensor.view()]?)?;

    let output = outputs["output"].try_extract_tensor::<f32>()?;

    let text = ctc_decode(output.view().as_slice().unwrap());

    // Write result back
    stream.write_all(text.as_bytes()).await?;
    stream.shutdown().await?;

    Ok(())
}

fn preprocess(data: &[u8]) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;

    let gray = img.to_luma8();
    let resized = image::imageops::resize(&gray, 200, 40, image::imageops::FilterType::Lanczos3);

    let normalized: Vec<f32> = resized.iter().map(|&p| (p as f32 / 255.0 - 0.5) / 0.5).collect();

    Ok(Array4::from_shape_vec((1, 1, 40, 200), normalized)?)
}

fn ctc_decode(output: &[f32]) -> String {
    let vocab_size = 37;
    let time_steps = 50;
    let mut result = String::new();
    let mut prev_idx = 0usize;

    for t in 0..time_steps {
        let start = t * vocab_size;
        let logits = &output[start..start + vocab_size];

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if idx != 0 && idx != prev_idx {
            if let Some(c) = CHARS.chars().nth(idx - 1) {
                result.push(c);
            }
        }
        prev_idx = idx;
    }

    result.to_uppercase()
}
