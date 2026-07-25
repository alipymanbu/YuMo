/**
 * Pipeline state constants shared across all frontend windows/pages.
 */

// --- State values -----------------------------------------------------------
export const PIPELINE_IDLE = 'idle' as const;
export const PIPELINE_RECORDING = 'recording' as const;
export const PIPELINE_PROCESSING = 'processing' as const;
export const PIPELINE_FILTERING = 'filtering' as const;
export const PIPELINE_TRANSCRIBING = 'transcribing' as const;
export const PIPELINE_PASTING = 'pasting' as const;

export type PipelineState =
  | typeof PIPELINE_IDLE
  | typeof PIPELINE_RECORDING
  | typeof PIPELINE_FILTERING
  | typeof PIPELINE_PROCESSING
  | typeof PIPELINE_TRANSCRIBING
  | typeof PIPELINE_PASTING;

// --- Tauri event / command names --------------------------------------------

export const EVENT_RECORDING_STATE = 'recording-state';
export const CMD_GET_PIPELINE_STATE = 'get_pipeline_state';

// --- Display labels (strategy map) ------------------------------------------

export const PIPELINE_LABELS: Record<PipelineState, string> = {
  [PIPELINE_RECORDING]: '录音中',
  [PIPELINE_FILTERING]: '声纹过滤中...',
  [PIPELINE_PROCESSING]: '处理中...',
  [PIPELINE_TRANSCRIBING]: '转录中...',
  [PIPELINE_PASTING]: '粘贴中...',
  [PIPELINE_IDLE]: '',
};

export const PIPELINE_LABEL_KEYS: Record<PipelineState, string> = {
  [PIPELINE_RECORDING]: 'pipeline.recording',
  [PIPELINE_FILTERING]: 'pipeline.filtering',
  [PIPELINE_PROCESSING]: 'pipeline.processing',
  [PIPELINE_TRANSCRIBING]: 'pipeline.transcribing',
  [PIPELINE_PASTING]: 'pipeline.pasting',
  [PIPELINE_IDLE]: '',
};

export const COLOR_ACTIVE = '#ff4d4f';
export const COLOR_PROCESSING = '#1890ff';

export const PIPELINE_COLORS: Record<PipelineState, string> = {
  [PIPELINE_RECORDING]: COLOR_ACTIVE,
  [PIPELINE_FILTERING]: COLOR_PROCESSING,
  [PIPELINE_PROCESSING]: COLOR_PROCESSING,
  [PIPELINE_TRANSCRIBING]: COLOR_PROCESSING,
  [PIPELINE_PASTING]: COLOR_PROCESSING,
  [PIPELINE_IDLE]: COLOR_PROCESSING,
};

export const PIPELINE_ANIMATIONS: Record<PipelineState, string> = {
  [PIPELINE_RECORDING]: 'pulse 1.5s infinite',
  [PIPELINE_FILTERING]: 'none',
  [PIPELINE_PROCESSING]: 'none',
  [PIPELINE_TRANSCRIBING]: 'none',
  [PIPELINE_PASTING]: 'none',
  [PIPELINE_IDLE]: 'none',
};

export function parsePipelineState(raw: string | undefined | null): PipelineState {
  const valid: ReadonlySet<string> = new Set([
    PIPELINE_IDLE,
    PIPELINE_RECORDING,
    PIPELINE_FILTERING,
    PIPELINE_PROCESSING,
    PIPELINE_TRANSCRIBING,
    PIPELINE_PASTING,
  ]);
  return valid.has(raw ?? '') ? (raw as PipelineState) : PIPELINE_IDLE;
}

/** Frame ranges for sprite animation per pipeline state.
 *  [startFrame, endFrame] inclusive. `null` or [0,0] = static first frame.
 *  Default assumes a 9-frame 3×3 sprite sheet:
 *    Row 1 (frames 0-2): idle, Row 2 (3-5): recording, Row 3 (6-8): processing
 */
export const PIPELINE_SPRITE_FRAMES: Record<PipelineState, [number, number] | null> = {
  [PIPELINE_IDLE]: [0, 0],
  [PIPELINE_RECORDING]: [0, 8],
  [PIPELINE_FILTERING]: [6, 8],
  [PIPELINE_PROCESSING]: [3, 5],
  [PIPELINE_TRANSCRIBING]: [3, 5],
  [PIPELINE_PASTING]: [6, 8],
};
// --- Helpers ----------------------------------------------------------------

