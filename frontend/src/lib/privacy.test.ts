import { describe, expect, it } from "vitest";
import loopbackEndpoints from "../test/loopback-endpoints.json";
import { isLocalEndpoint } from "./privacy";

describe("isLocalEndpoint", () => {
  it.each(loopbackEndpoints)("classifies $url consistently with the backend", ({ url, loopback }) => {
    expect(isLocalEndpoint({ use_llm: true, llm_url: url })).toBe(loopback);
  });

  it("is local when model assistance is disabled or has no endpoint", () => {
    expect(isLocalEndpoint({ use_llm: false, llm_url: "https://example.com" })).toBe(true);
    expect(isLocalEndpoint({ use_llm: true, llm_url: "" })).toBe(true);
  });
});
