/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as account from "../account.js";
import type * as apiKeys from "../apiKeys.js";
import type * as auth from "../auth.js";
import type * as bookmarks from "../bookmarks.js";
import type * as config from "../config.js";
import type * as crons from "../crons.js";
import type * as crypto from "../crypto.js";
import type * as dashboard from "../dashboard.js";
import type * as defaults from "../defaults.js";
import type * as evaluation from "../evaluation.js";
import type * as feed from "../feed.js";
import type * as feedback from "../feedback.js";
import type * as goldData from "../goldData.js";
import type * as http from "../http.js";
import type * as items from "../items.js";
import type * as labels from "../labels.js";
import type * as learning from "../learning.js";
import type * as mlops from "../mlops.js";
import type * as notifications from "../notifications.js";
import type * as pipeline from "../pipeline.js";
import type * as pipelineStore from "../pipelineStore.js";
import type * as prefs from "../prefs.js";
import type * as seedData from "../seedData.js";
import type * as sources from "../sources.js";
import type * as users from "../users.js";
import type * as websub from "../websub.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  account: typeof account;
  apiKeys: typeof apiKeys;
  auth: typeof auth;
  bookmarks: typeof bookmarks;
  config: typeof config;
  crons: typeof crons;
  crypto: typeof crypto;
  dashboard: typeof dashboard;
  defaults: typeof defaults;
  evaluation: typeof evaluation;
  feed: typeof feed;
  feedback: typeof feedback;
  goldData: typeof goldData;
  http: typeof http;
  items: typeof items;
  labels: typeof labels;
  learning: typeof learning;
  mlops: typeof mlops;
  notifications: typeof notifications;
  pipeline: typeof pipeline;
  pipelineStore: typeof pipelineStore;
  prefs: typeof prefs;
  seedData: typeof seedData;
  sources: typeof sources;
  users: typeof users;
  websub: typeof websub;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
