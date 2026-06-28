import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Convex codegen output.
    "convex/_generated/**",
  ]),
  {
    rules: {
      // Syncing Convex query results into local editable form state (sliders,
      // topic chips) is a legitimate external-store→state sync; keep it advisory.
      "react-hooks/set-state-in-effect": "warn",
    },
  },
]);

export default eslintConfig;
