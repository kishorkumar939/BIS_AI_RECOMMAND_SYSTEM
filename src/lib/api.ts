import type {
  BISRecommendation,
  LegalCitation,
  AuditResult,
  RecommendationResponse,
} from "@/types/bis";

export type { LegalCitation, AuditResult, BISRecommendation, RecommendationResponse };

// Clean and sanitize backend API base URL (trims accidental whitespace and trailing slashes)
const rawBase = (
  import.meta.env.VITE_API_BASE ||
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  "http://localhost:8000"
).trim();

const API_BASE = rawBase.replace(/\/+$/, "").replace(/\/api\/v1$/, "").replace(/\/api$/, "");

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchRecommendations(
  query: string,
  topK: number = 5,
  generateClause: boolean = true
): Promise<RecommendationResponse> {
  const res = await fetch(`${API_BASE}/api/v1/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, generate_clause: generateClause }),
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch recommendations from server (status: ${res.status})`);
  }

  const data = await res.json();
  return {
    recommendations: data.recommendations || [],
    compliance_clause: data.compliance_clause || null,
    legal_framework: data.legal_framework || [],
  };
}

export async function auditPDF(file: File): Promise<AuditResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/v1/audit-pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Failed to audit PDF document (status: ${res.status})`);
  }

  return await res.json();
}
