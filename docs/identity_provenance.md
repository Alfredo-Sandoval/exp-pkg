# Identity Provenance

`xpkg` should record where multi-animal identities came from. It should not
train, run, or claim a ReID model.

## Design Direction

Keep `Track` as the lightweight pose-track identity used by labels. Store
identity provenance as companion records keyed by track id or track name rather
than widening every `Track` field up front.

The first contract should answer:

- whether identity came from MOT, ReID, manual assignment, mixed sources, or an
  unknown source
- which upstream tool/file produced the track identity
- where identity confidence applies by frame span
- where identity swaps were detected or manually corrected
- which spans were manually proofread

## Suggested Record Shape

A companion identity-provenance payload should use simple JSON records:

```json
{
  "track_id": "track_0",
  "source_tool": "sleap",
  "source_file": "analysis.h5",
  "identity_source": "mot",
  "spans": [
    {
      "video_id": "video_0",
      "start_frame": 0,
      "end_frame": 100,
      "identity_source": "mot",
      "confidence": 0.92
    }
  ],
  "events": [
    {
      "kind": "identity_swap",
      "video_id": "video_0",
      "frame": 101,
      "from_track_id": "track_0",
      "to_track_id": "track_1"
    }
  ],
  "proofreading": [
    {
      "video_id": "video_0",
      "start_frame": 0,
      "end_frame": 250,
      "reviewed": true,
      "corrected": false,
      "reviewer": "manual"
    }
  ]
}
```

Use `unknown` when an importer cannot distinguish MOT from ReID. Do not infer a
stronger source than the upstream file exposes.

## Importer Implication

SLEAP and DLC multi-animal importers should populate this companion provenance
only with facts available in the source export. SLEAP track names can map to
track ids, but a source file that does not state ReID versus MOT should remain
`unknown` or tool-specific metadata.

DLC multi-animal parsing is a separate importer task. Do not bundle it into the
first identity-provenance model change.
