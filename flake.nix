{
  description = "3D AST Visualizer — Bevy + WebGL/WASM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # crane: Rust+Nix builder with first-class trunk support. Vendors
    # cargo deps from Cargo.lock so `nix build` works inside the sandbox
    # (no crates.io reachability needed at build time).
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, crane, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        inherit (pkgs) lib;

        # Rust toolchain with WASM target
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
          extensions = [ "rust-analyzer" "rust-src" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # ── Bevy native runtime deps (Linux) ──────────────
        # Used by the devShell for `cargo run` (desktop). The WASM build
        # doesn't need these — browsers don't link against alsa/x11.
        bevyNativeDeps = with pkgs; [
          # X11
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          # Wayland
          libxkbcommon
          wayland
          # Audio
          alsa-lib
          # Input / udev
          udev
          # Rendering
          vulkan-loader
          libGL
        ];

        # ── Build-time deps for devShell ──────────────────
        buildDeps = with pkgs; [
          pkg-config
          cmake       # some transitive crates need this
          perl        # openssl-sys sometimes pulls this in
          trunk       # WASM bundler
          wasm-bindgen-cli
          mold        # fast linker (used by .cargo/config.toml)
          clang       # linker driver for mold
        ];

        # Cargo.lock pins wasm-bindgen = 0.2.114. The wasm-bindgen-cli
        # used to produce the JS glue must speak the exact same schema
        # version — mismatch → the loaded WASM panics on boot with
        # "wasm-bindgen output v_ does not match generated v_". nixpkgs
        # ships a different version, so we build the CLI ourselves at
        # 0.2.114 via crane.
        #
        # We can't use `pkgs.buildWasmBindgenCli` here: it pulls deps via
        # `rustPlatform.fetchCargoVendor`, whose Python downloader gets
        # 403'd by crates.io for ~random crates (UA / rate-limit issue,
        # nixpkgs side). Crane uses curl per-crate and doesn't trip it.
        wasmBindgenCli = craneLib.buildPackage {
          pname = "wasm-bindgen-cli";
          version = "0.2.114";
          src = pkgs.fetchCrate {
            pname = "wasm-bindgen-cli";
            version = "0.2.114";
            hash = "sha256-xrCym+rFY6EUQFWyWl6OPA+LtftpUAE5pIaElAIVqW0=";
          };
          doCheck = false;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
        };

        # cleanCargoSource would strip index.html (trunk entry point) and
        # assets/ (shaders pulled in via include_str! at compile time).
        # Extend the filter to keep both.
        src = lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type)
            || (baseNameOf path == "index.html")
            || (lib.hasInfix "/assets/" path);
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ rustToolchain ] ++ buildDeps ++ bevyNativeDeps;

          # ── Critical: tell the linker where to find .so files ──
          # Without this, Bevy (native) fails at link time or runtime
          # because NixOS doesn't have /usr/lib.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath bevyNativeDeps;

          # pkg-config needs to find alsa, udev, x11, etc.
          PKG_CONFIG_PATH = pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" bevyNativeDeps;

          shellHook = ''
            echo ""
            echo "  🌳 AST Visualizer 3D — dev shell"
            echo ""
            echo "  WASM (browser):  trunk serve --release"
            echo "  Native (desktop): cargo run --release"
            echo ""
          '';
        };

        # ── WASM build package ────────────────────────────
        # `nix build` produces the dist/ output for deployment.
        #
        # Crane vendors all cargo deps from Cargo.lock into the Nix store
        # so the build is fully offline. wasm-bindgen-cli is pre-pinned
        # (above) and wired into trunk's tool cache by crane — trunk
        # therefore won't try to download anything during the build.
        packages.default = craneLib.buildTrunkPackage {
          inherit src;
          pname = "bevy-ast-3d";
          version = "0.1.0";

          # Build cargo against wasm32 — Bevy's native runtime deps are
          # irrelevant for the browser target.
          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
          doCheck = false;

          # Pin trunk's wasm-bindgen to the schema version Cargo.lock expects.
          wasm-bindgen-cli = wasmBindgenCli;
        };
        # Crane adds `--release=true` automatically because CARGO_PROFILE
        # defaults to "release" — no need for an extra --release flag.
      });
}
