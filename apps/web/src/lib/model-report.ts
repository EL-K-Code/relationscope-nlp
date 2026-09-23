import type { OpenSetProfile } from "@/lib/types";

export const FINAL_METRICS = {
  knownMacroF1: 0.8915730905612583,
  knownAccuracy: 0.8920918367346938,
  knownTop3: 0.9801020408163266,
  knownEce: 0.019492929272961864,
  oodAuRoc: 0.8032652036260932,
  unknownF1: 0.8529557649817379,
  knownAcceptance: 0.6125,
  unknownRejection: 0.8444642857142857,
  warmCpuMedianMs: 87.2,
  warmClientMedianMs: 138.0,
} as const;

export const PROFILE_METRICS: Record<
  OpenSetProfile,
  {
    label: string;
    shortLabel: string;
    threshold: number;
    knownAcceptance: number;
    acceptedKnownAccuracy: number;
    unknownRejection: number;
    unknownF1: number;
    description: string;
  }
> = {
  high_coverage: {
    label: "High coverage",
    shortLabel: "Coverage",
    threshold: 0.924,
    knownAcceptance: 0.724,
    acceptedKnownAccuracy: 0.9767,
    unknownRejection: 0.7379,
    unknownF1: 0.8044,
    description: "Accepts more known examples while keeping calibrated abstention.",
  },
  balanced: {
    label: "Balanced",
    shortLabel: "Balanced",
    threshold: 0.967,
    knownAcceptance: 0.6125,
    acceptedKnownAccuracy: 0.9867,
    unknownRejection: 0.8445,
    unknownF1: 0.853,
    description: "Default operating point balancing coverage and unseen-relation rejection.",
  },
  high_safety: {
    label: "High safety",
    shortLabel: "Safety",
    threshold: 0.978,
    knownAcceptance: 0.5543,
    acceptedKnownAccuracy: 0.9894,
    unknownRejection: 0.8863,
    unknownF1: 0.868,
    description: "Abstains more aggressively when confidence is not high enough.",
  },
};

export const RISK_COVERAGE = [
  { coverage: 1.0, risk: 0.108 },
  { coverage: 0.9, risk: 0.062 },
  { coverage: 0.8, risk: 0.034 },
  { coverage: 0.7, risk: 0.019 },
  { coverage: 0.6, risk: 0.013 },
  { coverage: 0.5, risk: 0.009 },
] as const;

export const EVALUATION_PROTOCOL = {
  name: "fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown",
  knownRelations: 56,
  calibrationUnknownRelations: 8,
  finalUnknownRelations: 16,
  trainKnown: 31360,
  validationKnown: 3920,
  testKnown: 3920,
  calibrationUnknown: 5600,
  testUnknown: 11200,
} as const;
