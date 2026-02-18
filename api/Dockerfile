# Build stage
FROM alpine:latest AS builder

# Install build dependencies, rust, cargo, and onnxruntime development headers
RUN apk add --no-cache musl-dev gcc pkgconfig onnxruntime-dev g++ make rust cargo

WORKDIR /app
COPY . .

# Use system onnxruntime to avoid downloading incompatible glibc binaries
ENV ORT_STRATEGY=system
RUN cargo build --release

# Runtime stage
FROM alpine:latest

# Install onnxruntime and other runtime dependencies
RUN apk add --no-cache libgcc libstdc++ onnxruntime

WORKDIR /app

# Copy binaries
COPY --from=builder /app/target/release/api /app/api
COPY --from=builder /app/target/release/solver /app/solver

# Copy model assets
COPY model/captcha_model.onnx /app/model/captcha_model.onnx
COPY model/captcha_model.onnx.data /app/model/captcha_model.onnx.data

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default port
ENV PORT=3000

ENTRYPOINT ["/app/entrypoint.sh"]
