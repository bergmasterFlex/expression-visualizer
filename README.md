# AST Visualizer 3D — Bevy + WebGL

A 3D Abstract Syntax Tree visualizer for arithmetic expressions with ternary
operator support, built with **Rust** and **Bevy**, compiled to **WebAssembly**
and rendered via **WebGL2**.

![screenshot](https://img.shields.io/badge/Bevy-0.14-blue)
![screenshot](https://img.shields.io/badge/Rust-2021-orange)
![screenshot](https://img.shields.io/badge/Target-WebGL2-green)

## Features

- **Live parsing** — type expressions and watch the 3D tree rebuild
- **Evaluation-order layout** — Y axis represents evaluation order (bottom =
  first, top = result)
- **Z-axis ternary branching** — `then`/`else` branches split orthogonally to
  binary splits, flowing *upward* from the `?:` node
- **Rich node types** — spheres (numbers), cubes (booleans), octahedrons
  (ternary), color-coded operators
- **Orbit camera** — drag to rotate, scroll to zoom, right-drag to pan,
  auto-rotation on idle
- **Curved edge tubes** — bezier-curved gizmo edges with directional arrows

## Supported syntax

| Feature    | Examples                            |
|------------|-------------------------------------|
| Numbers    | `1`, `3.14`, `42`                   |
| Booleans   | `true`, `false`                     |
| Arithmetic | `+`, `-`, `*`, `/`                  |
| Comparison | `>`, `<`, `>=`, `<=`, `==`, `!=`    |
| Ternary    | `cond ? then : else`                |
| Grouping   | `(expr)`                            |

## Prerequisites

1. **Rust** (stable, 1.77+):
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **WASM target**:
   ```sh
   rustup target add wasm32-unknown-unknown
   ```

3. **Trunk** (WASM build tool):
   ```sh
   cargo install trunk
   ```

## Build & Run

### WebGL (browser)

```sh
trunk serve --release
```

Then open [http://localhost:8080](http://localhost:8080).

For a production build:

```sh
trunk build --release
```

Output is in `dist/`.

### Desktop (native, for development)

```sh
cargo run --release
```

> Note: When running natively, the expression is fixed to the default since
> the HTML/JS bridge is only active in WASM mode. To change expressions
> natively, modify `AstState::default()` in `src/main.rs`.

## Architecture

```
src/
├── main.rs      Bevy app setup, systems, rendering, WASM bridge
├── ast.rs       Tokenizer + recursive-descent parser
├── layout.rs    Evaluation-order 3D layout engine
└── camera.rs    Orbit camera controller plugin
index.html       Trunk entry point + HTML UI overlay
```

**Design decisions:**

- **UI split**: The expression input, presets, and legend live in HTML/CSS
  (overlaid on the canvas). This avoids the complexity of text input widgets in
  Bevy and gives a crisp, responsive UI. The WASM module polls
  `window.astExpression` each frame via `web-sys`.

- **Gizmos for edges**: Bevy's `Gizmos` system draws the bezier-curved edges
  each frame. This is simpler and more flexible than spawning tube meshes, and
  the line rendering works well in WebGL2.

- **Evaluation-order Y**: The layout algorithm assigns Y positions based on
  data flow. Leaf nodes (literals) are at the bottom; the final result-producing
  node is at the top. For ternary, the condition sits below, and the
  then/else branches fan out above along the Z axis.

## License

MIT
