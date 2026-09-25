# Review Branch (`active_learning/review/`)

The review tool lets you quickly inspect model detections on raw frames, fix false positives, adjust misaligned bounding boxes, and label false negatives.

## Key Controls

| Action | Control |
|---|---|
| **Draw Box** | Left-drag on empty space |
| **Move Box** | Left-drag inside existing box |
| **Resize Box** | Left-drag near any corner |
| **Delete Box** | Right-click box |
| **Active Class for New Boxes** | Number keys `0`–`9` |
| **Reclassify Existing Box** | Click box to highlight yellow, then press digit `0`–`9` |
| **Accept Frame** | `s` or `Enter` (saves image + YOLO label) |
| **Reject Frame** | `n` or `Space` (skips frame) |
| **Undo** | `z` |
| **Reset to Original** | `u` |
| **Quit & Save Progress** | `q` or `Esc` (persists state in `review_state.json`) |

## Quick Command Examples

```bash
# Review default raw dataset using base weights:
python review_app.py

# Review specific image folder with custom weights:
python review_app.py --images-dir ../dataset/raw --weights ../../weights/best.pt --output ../dataset/reviewed/batch_01

# Review using pre-computed YOLO .txt predictions:
python review_app.py --images-dir ../dataset/raw --preds-dir path/to/preds/
```
