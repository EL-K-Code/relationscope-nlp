import { GoogleAuth } from "google-auth-library";
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.RELATIONSCOPE_API_URL ?? "http://localhost:8080";
const BACKEND_AUDIENCE = process.env.RELATIONSCOPE_API_AUDIENCE;

type Context = {
  params: Promise<{ path: string[] }>;
};

async function authorizationHeader(): Promise<string | null> {
  if (!BACKEND_AUDIENCE) return null;

  const auth = new GoogleAuth();
  const client = await auth.getIdTokenClient(BACKEND_AUDIENCE);
  const token = await client.idTokenProvider.fetchIdToken(BACKEND_AUDIENCE);
  return `Bearer ${token}`;
}

async function forward(request: NextRequest, context: Context) {
  const { path } = await context.params;
  const target = new URL(path.join("/"), `${BACKEND_URL.replace(/\/$/, "")}/`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const authorization = await authorizationHeader();
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
          : await request.text(),
      cache: "no-store",
    });

    return new NextResponse(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
      },
    });
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
