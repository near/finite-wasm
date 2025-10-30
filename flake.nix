{
  nixConfig.extra-substituters = [
    "https://nixify.cachix.org"
    "https://nix-community.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "nixify.cachix.org-1:95SiUQuf8Ij0hwDweALJsLtnMyv/otZamWNRp1Q1pXw="
    "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    "cache.garnix.io:CTFPyKSLcx5RMJKfLo5EEPUObbA78b0YQ2DTCJXqr9g="
  ];

  inputs.nixify.inputs.nixlib.follows = "nixlib";
  inputs.nixify.url = "github:rvolosatovs/nixify";
  inputs.nixlib.url = "github:nix-community/nixpkgs.lib";

  outputs = {
    nixify,
    nixlib,
    ...
  }:
    with nixlib.lib;
    with nixify.lib;
      rust.mkFlake {
        src = ./.;

        withDevShells = {
          devShells,
          pkgs,
          ...
        }:
          extendDerivations {
            buildInputs = let
              ocamlPackages = pkgs.ocaml-ng.ocamlPackages_4_14;
            in [
              ocamlPackages.ocaml
              ocamlPackages.ocamlbuild
            ];
          }
          devShells;
      };
}
