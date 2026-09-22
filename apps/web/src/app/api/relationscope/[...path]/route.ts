import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.RELATIONSCOPE_API_URL ?? "http://localhost:8080";

type Context = {
  params: Promise<{ path: string[] }>;
};

async function forward(request: NextRequest, context: Context) {
  const { path } = await context.params;
  const target = new URL(path.join("/"), `${BACKEND_URL.replace(/\/$/, "")}/`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

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
}

export async function GET(request: NextRequest, context: Context) {
  return forward(request, context);
}

export async function POST(request: NextRequest, context: Context) {
  return forward(request, context);
}
