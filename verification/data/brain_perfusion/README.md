# Brain Perfusion Data

The large binary data files in this directory are **not tracked in git** (confidential phantom data).

## Where to get the files

Copy from the group share drive:

```
smb://160.87.12.113/Molloilab/Wenbo/brain phantom/Caedin Files/dynamic_brain_phantom
```

## Required files

| File | Size | Description |
|------|------|-------------|
| `P1_brain_all_2020_RAW_400_400_400.raw` | ~122 MB | Patient 1 XCAT brain phantom (400³ uint16) |
| `P2_brain_all_2020_RAW_400_400_400.raw` | ~122 MB | Patient 2 XCAT brain phantom (400³ uint16) |
| `iodine_mass_data.mat` | ~501 MB | Per-segment iodine mass from perfusion simulation |
| `structure_info.mat` | — | XCAT segment ID ↔ tissue name mapping |

The `.txt` table files (`P1_voxelize_table.txt`, `P2_vozelize_table.txt`) are small and are tracked in git.

## Connecting to the share drive

**macOS**: Finder → Go → Connect to Server → paste the `smb://` path above  
**Windows**: Map network drive → `\\160.87.12.113\Molloilab\Wenbo\brain phantom\Caedin Files\dynamic_brain_phantom`
