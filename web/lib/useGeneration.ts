"use client";

import { useCallback, useRef, useState } from "react";
import { generate } from "./api";
import type { GenerationParams, GenerationResult, Variant } from "./types";

export type GenStage = "idle" | "load" | "denoise" | "decode" | "done" | "error";

export interface GenState {
  stage: GenStage;
  message: string;
  cold: boolean;
  step: number;
  totalSteps: number;
  previewUrl?: string;
  result?: GenerationResult;
  error?: string;
  running: boolean;
}

const INITIAL: GenState = {
  stage: "idle",
  message: "",
  cold: false,
  step: 0,
  totalSteps: 0,
  running: false,
};

/** Drives a single generation slot (Studio uses one; Compare uses two). */
export function useGeneration() {
  const [state, setState] = useState<GenState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((s) => (s.running ? { ...s, running: false, stage: "idle", message: "Cancelled" } : s));
  }, []);

  const run = useCallback(async (params: GenerationParams, variant: Variant) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState({
      ...INITIAL,
      running: true,
      stage: "load",
      totalSteps: params.steps,
      message: "Queuing…",
    });

    try {
      await generate(params, variant, {
        signal: ctrl.signal,
        onEvent: (e) => {
          setState((prev) => {
            switch (e.type) {
              case "status":
                return { ...prev, stage: e.stage, message: e.message, cold: e.cold };
              case "progress":
                return {
                  ...prev,
                  stage: "denoise",
                  step: e.step,
                  totalSteps: e.totalSteps,
                  previewUrl: e.previewUrl ?? prev.previewUrl,
                };
              case "done":
                return {
                  ...prev,
                  stage: "done",
                  running: false,
                  result: e.result,
                  step: prev.totalSteps,
                  previewUrl: e.result.imageUrl,
                };
              case "error":
                return { ...prev, stage: "error", running: false, error: e.message };
            }
          });
        },
      });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setState((prev) => ({ ...prev, stage: "error", running: false, error: String(err) }));
    }
  }, []);

  const reset = useCallback(() => setState(INITIAL), []);

  return { state, run, cancel, reset };
}
