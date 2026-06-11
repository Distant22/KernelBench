#!/usr/bin/env bash
set -euo pipefail

# Installs standalone NVIDIA Nsight CLI packages under the repository only.
# Supply official NVIDIA installer URLs because NVIDIA changes release filenames.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.tools/nsight"
DOWNLOADS="${DEST}/downloads"
mkdir -p "${DOWNLOADS}" "${DEST}/bin"

if [[ -z "${NSYS_INSTALLER_URL:-}" || -z "${NCU_INSTALLER_URL:-}" ]]; then
  cat <<'EOF'
Set both official NVIDIA standalone installer URLs, then rerun:

  NSYS_INSTALLER_URL=<Nsight Systems .run URL> \
  NCU_INSTALLER_URL=<Nsight Compute .run URL> \
  bash finalProject_260531/install_nsight_tools.sh

The script installs only under .tools/nsight and does not use sudo.
EOF
  exit 2
fi

download_and_install() {
  local name="$1"
  local url="$2"
  local installer="${DOWNLOADS}/${name}.run"
  curl --fail --location "${url}" --output "${installer}"
  chmod +x "${installer}"
  "${installer}" --quiet --targetpath="${DEST}/${name}"
}

download_and_install nsys "${NSYS_INSTALLER_URL}"
download_and_install ncu "${NCU_INSTALLER_URL}"

ln -sfn "$(find "${DEST}/nsys" -type f -name nsys -perm -u+x | head -1)" "${DEST}/bin/nsys"
ln -sfn "$(find "${DEST}/ncu" -type f -name ncu -perm -u+x | head -1)" "${DEST}/bin/ncu"

echo "Installed repository-local tools:"
"${DEST}/bin/nsys" --version
"${DEST}/bin/ncu" --version
