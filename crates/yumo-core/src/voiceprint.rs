//! Voiceprint (声纹) worker invocation.
//!
//! Runs `voiceprint_worker.py` as a one-shot child process per action,
//! independent of the transcription daemon. Matches the `custom_model_worker.py`
//! protocol: stdin JSON command → stdout last-line JSON response.
//!
//! Protocol:
//!   stdin  — a single JSON command line
//!   stdout — exactly one JSON object: {"ok": true, ...} or {"ok": false, "error": "..."}
//!   stderr — log lines and (on failure) python traceback
//!   exit   — 0 on success, non-zero on failure

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::error::{AppError, AppResult};

pub type WorkerResponse = serde_json::Map<String, Value>;

/// Spawn the voiceprint worker, send `cmd`, await its single JSON reply.
pub async fn run_voiceprint_action(
    python: &str,
    worker_script: &Path,
    cmd: Value,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let action = cmd
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    log::info!("[voiceprint] action={action} starting (timeout={timeout:?})");

    let cmd_json = serde_json::to_string(&cmd)
        .map_err(|e| AppError::Transcription(format!("serialize voiceprint cmd: {e}")))?;

    let cwd = worker_script.parent().ok_or_else(|| {
        AppError::Transcription(format!(
            "voiceprint worker script has no parent dir: {}",
            worker_script.display()
        ))
    })?;

    let mut child = Command::new(python)
        .arg(worker_script)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            AppError::Transcription(format!("spawn voiceprint worker: {e}"))
        })?;

    // Write command to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(cmd_json.as_bytes())
            .await
            .map_err(|e| {
                AppError::Transcription(format!("write voiceprint stdin: {e}"))
            })?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| {
                AppError::Transcription(format!("write voiceprint stdin newline: {e}"))
            })?;
        drop(stdin);
    }

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return Err(AppError::Transcription(format!(
                "voiceprint worker wait: {e}"
            )));
        }
        Err(_) => {
            // Timeout — child already consumed by wait_with_output
            return Err(AppError::Transcription(format!(
                "voiceprint worker action={action} timed out after {timeout:?}"
            )));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stderr.is_empty() && !stderr.trim().is_empty() {
        log::info!("[voiceprint] stderr ({} bytes): {}", stderr.len(), stderr.trim_end());
    }

    // Worker contract: last non-empty stdout line is the JSON reply
    let last_line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|s| s.trim().to_string());

    let resp = match last_line {
        Some(line) => line,
        None => {
            let snippet = if stdout.len() > 500 {
                &stdout[..500]
            } else {
                &stdout
            };
            return Err(AppError::Transcription(format!(
                "voiceprint worker stdout empty. stdout({}b)=\"{}\" stderr=\"{}\"",
                stdout.len(),
                snippet,
                stderr.trim_end()
            )));
        }
    };

    let resp_obj: WorkerResponse = serde_json::from_str(&resp).map_err(|e| {
        AppError::Transcription(format!(
            "voiceprint worker parse JSON: {e}. raw=\"{resp}\" stderr=\"{}\"",
            stderr.trim_end()
        ))
    })?;

    let ok = resp_obj
        .get("ok")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !ok {
        let error_msg = resp_obj
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown voiceprint error");
        return Err(AppError::Transcription(format!(
            "voiceprint worker action={action} error: {error_msg}. stderr=\"{}\"",
            stderr.trim_end()
        )));
    }

    if !output.status.success() {
        log::warn!(
            "[voiceprint] action={action} exit code non-zero but ok=true. stderr=\"{}\"",
            stderr.trim_end()
        );
    }

    log::info!("[voiceprint] action={action} completed");
    Ok(resp_obj)
}

// ---------------------------------------------------------------------------
// Thin wrappers — paths passed by caller
// ---------------------------------------------------------------------------

/// Ensure voiceprint models (Silero VAD + CAM++) are downloaded.
pub async fn ensure_models(
    python: &str,
    worker: &Path,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let cmd = serde_json::json!({"action": "ensure_models"});
    run_voiceprint_action(python, worker, cmd, timeout).await
}

/// Enroll speaker profile from history recordings.
pub async fn enroll_from_history(
    python: &str,
    worker: &Path,
    recordings_dir: &str,
    max_files: u32,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let cmd = serde_json::json!({
        "action": "enroll_from_history",
        "recordings_dir": recordings_dir,
        "max_files": max_files,
    });
    run_voiceprint_action(python, worker, cmd, timeout).await
}

/// Filter audio through voiceprint gate.
/// `input` and `output` are absolute WAV file paths.
pub async fn filter_wav(
    python: &str,
    worker: &Path,
    input: &str,
    output: &str,
    threshold: f64,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let cmd = serde_json::json!({
        "action": "filter",
        "audio": input,
        "output": output,
        "threshold": threshold,
    });
    run_voiceprint_action(python, worker, cmd, timeout).await
}

/// Query voiceprint status (enrolled? models ready?).
pub async fn status(
    python: &str,
    worker: &Path,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let cmd = serde_json::json!({"action": "status"});
    run_voiceprint_action(python, worker, cmd, timeout).await
}

/// Clear enrolled profile (keep models).
pub async fn clear(
    python: &str,
    worker: &Path,
    timeout: Duration,
) -> AppResult<WorkerResponse> {
    let cmd = serde_json::json!({"action": "clear"});
    run_voiceprint_action(python, worker, cmd, timeout).await
}
