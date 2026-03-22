{
  description = "3D AST Visualizer — Bevy + WebGL/WASM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Rust toolchain with WASM target
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
          extensions = [ "rust-analyzer" "rust-src"];
        };

        # ── Bevy native runtime deps (Linux) ──────────────
        # Bevy links against these dynamically at runtime.
        # Without them you get missing .so errors on `cargo run`.
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

        # ── Build-time deps ───────────────────────────────
        buildDeps = with pkgs; [
          pkg-config
          cmake       # some transitive crates need this
          perl        # openssl-sys sometimes pulls this in
          trunk       # WASM bundler
          wasm-bindgen-cli
          mold        # fast linker (used by .cargo/config.toml)
          clang       # linker driver for mold
        ];

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
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "bevy-ast-3d";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ rustToolchain pkgs.trunk pkgs.wasm-bindgen-cli pkgs.pkg-config ];
          buildInputs = bevyNativeDeps;

          # Trunk needs a writable cargo home and CARGO_TARGET_DIR
          buildPhase = ''
            export HOME=$(mktemp -d)
            export CARGO_HOME=$HOME/.cargo
            trunk build --release
          '';

          installPhase = ''
            cp -r dist $out
          '';
        };
      }
    );
}
