import { GoogleAuth } from "google-auth-library";
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.RELATIONSCOPE_API_URL ?? "http://localhost:8080";
const BACKEND_AUDIENCE = process.env.RELATIONSCOPE_API_AUDIENCE;
const CHALLENGER_URL = process.env.RELATIONSCOPE_CHALLENGER_API_URL;
const CHALLENGER_AUDIENCE =
  process.env.RELATIONSCOPE_CHALLENGER_API_AUDIENCE;

function canaryPercent() {
  const value = Number(process.env.RELATIONSCOPE_CANARY_PERCENT ?? "0");
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

type Context = {
  params: Promise<{ path: string[] }>;
};

function stableBucket(key: string) {
  let hash = 2166136261;
  for (let index = 0; index < key.length; index += 1) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % 10_000;
}

function canaryEligible(path: string, method: string) {
  if (method !== "POST") return false;
  return (
    path === "v2/relations:predict-open-set" ||
    path === "v2/relations:explain" ||
    path === "v3/documents:analyze" ||
    path === "v3/documents:upload"
  );
}

async function authorizationHeader(
  audience?: string,
): Promise<string | null> {
  if (!audience) return null;

  const auth = new GoogleAuth();
  const client = await auth.getIdTokenClient(audience);
  const token = await client.idTokenProvider.fetchIdToken(audience);
  return `Bearer ${token}`;
}

async function forward(request: NextRequest, context: Context) {
  const { path } = await context.params;
  const joinedPath = path.join("/");

  // Internal callbacks and review/admin routes are IAM-only and are never
  // surfaced by the public Next.js proxy.
  if (
    joinedPath.startsWith("v2/internal/") ||
    joinedPath.startsWith("v3/internal/")
  ) {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }

  const existingCanaryKey = request.cookies.get(
    "relationscope-canary-key",
  )?.value;
  const canaryKey = existingCanaryKey ?? crypto.randomUUID();
  const percent = canaryPercent();
  const useChallenger =
    Boolean(CHALLENGER_URL) &&
    canaryEligible(joinedPath, request.method) &&
    stableBucket(canaryKey) < percent * 100;

  const baseUrl = useChallenger ? CHALLENGER_URL! : BACKEND_URL;
  const audience = useChallenger
    ? CHALLENGER_AUDIENCE
    : BACKEND_AUDIENCE;
  const target = new URL(joinedPath, `${baseUrl.replace(/\/$/, "")}/`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const authorization = await authorizationHeader(audience);
  if (authorization) {
    headers.set("authorization", authorization);
  }

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body:
        request.method === "GET" || request.method === "HEAD"
          ? undefined
          : await request.arrayBuffer(),
      cache: "no-store",
    });

    const proxyResponse = new NextResponse(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
        "x-relationscope-variant": useChallenger
          ? "challenger"
          : "champion",
      },
    });

    if (!existingCanaryKey) {
      proxyResponse.cookies.set("relationscope-canary-key", canaryKey, {
        httpOnly: true,
        sameSite: "lax",
        secure: process.env.NODE_ENV === "production",
        maxAge: 60 * 60 * 24 * 30,
        path: "/",
      });
    }

    return proxyResponse;
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "RelationScope API unavailable";
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}

export async function GET(request: NextRequest, context: Context) {
  return forward(request, context);
}

export async function POST(request: NextRequest, context: Context) {
  return forward(request, context);
}
