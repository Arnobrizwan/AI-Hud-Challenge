import { httpRouter } from "convex/server";
import { auth } from "./auth";

const http = httpRouter();

// Mounts Convex Auth's HTTP routes (sign-in, OAuth callbacks, token refresh).
auth.addHttpRoutes(http);

export default http;
