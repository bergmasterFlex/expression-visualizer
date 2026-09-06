# Expression Visualizer 3D — Bevy

A 3D editor for expression term graphs: nodes are placed on a grid whose axes
carry meaning — evaluation order, conjunction, disjunction — and edges show the
types travelling along them. Built with **Rust** and **Bevy**, compiled to
**Native** or **WebAssembly** with **WebGPU**.

## Features
Todo


## Build & Run

### WebGPU (browser)

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

**Design decisions:**
Todo

## License

MIT
