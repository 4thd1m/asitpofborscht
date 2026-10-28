// Carries the Rust tools that netlify.toml's build command installs from one
// deploy to the next, so they are compiled once instead of on every build.
// TOOLS_HOME must match the build command's TOOLS_HOME. Netlify runs onPreBuild
// before the build command and onPostBuild only after it succeeds, so a failed
// build never caches a half-installed tool.
const os = require("os");
const path = require("path");

const TOOLS_HOME = path.join(os.homedir(), ".sitp-tools");

module.exports = {
  onPreBuild: async ({ utils }) => {
    const restored = await utils.cache.restore(TOOLS_HOME);
    console.log(restored ? `restored ${TOOLS_HOME}` : `no cached ${TOOLS_HOME} yet`);
  },
  onPostBuild: async ({ utils }) => {
    await utils.cache.save(TOOLS_HOME);
  },
};
