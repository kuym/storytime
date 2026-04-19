// swift-tools-version: 5.12
import PackageDescription

let package = Package(
    name: "KokoroMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "KokoroMLX", type: .static, targets: ["KokoroMLX"]),
    ],
    dependencies: [
        .package(path: "../vendor/mlx-swift"),
    ],
    targets: [
        .target(
            name: "KokoroMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            path: "Sources/KokoroMLX"
        ),
    ]
)
