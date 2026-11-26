# Data: CHILDES transcripts used by this project

All transcripts in `data/childes/` come from **CHILDES**, part of the TalkBank system
(https://talkbank.org/childes/), downloaded with a free TalkBank account. They are de-identified
transcripts of children's speech in CHAT format (`.cha`). No audio or video is included.

## License and rules (please read before using or sharing)

- **License: Creative Commons CC BY-NC-SA 4.0** (TalkBank Ground Rules, https://talkbank.org/0share/rules.html).
  - **BY:** cite TalkBank and each corpus you use (see below).
  - **NC:** non-commercial use only. This also excludes using the data to train commercial
    systems such as large language models.
  - **SA:** anything derived from these files must be shared under the same license.
- Follow the **TalkBank Code of Ethics**. In particular: **never try to identify any child or family.**
  These are minors' transcripts. Some headers contain birthdates, test locations, examiner
  names or ethnicity, exactly as TalkBank distributes them; do not combine them with other
  information to identify anyone.
- Only public, non-password-protected CHILDES corpora are included. Password-protected TalkBank
  banks (AphasiaBank, DementiaBank, FluencyBank, RHDBank, TBIBank), controlled-access corpora,
  and all media files are **not** included and must never be added here.
- The canonical, most up-to-date copies live on TalkBank. If you publish anything, cite and
  link TalkBank as the source.

## General citation (required)

MacWhinney, B. (2000). *The CHILDES Project: Tools for Analyzing Talk* (3rd ed.). Lawrence Erlbaum Associates.

Each corpus also has its own required citation, listed on its TalkBank corpus page
(https://talkbank.org/childes/ → browse by language group). Copy the exact citation from
that page into any write-up.

## Corpora and how the project uses them

| Folder | CHILDES section | Language | Role in this project |
|---|---|---|---|
| `Brown/` | Eng-NA | English | Typical-development norms (3 children, 214 sessions, 1.5-5 yr) |
| `Bates/` | Eng-NA | English | Typical-development norms (toddlers, 20 and 28 mo) |
| `Gleason/` | Eng-NA | English | Typical-development norms (2-5 yr) |
| `ENNI/` | Clinical-Eng | English | Delay model training and testing (345 children, 4-9 yr, TD + SLI) |
| `Gillam/` | Clinical-Eng | English | Outside test set, never trained on (664 children, 5-11 yr, TD + SLI) |
| `Conti4/` | Clinical-Eng | English (UK) | Outside test set, never trained on (118 teens, 13-16 yr, TD + SLI) |
| `Szagun/` | German | German | Second-language language-age check (TD) and delay check (CI) |
| `HKU/` | Chinese (Cantonese) | Cantonese | Second-language language-age check |
| `Ornat/`, `Marrero/`, `FernAguado/` | Spanish | Spanish | Spanish premise check (data-limited) |
| `Aksu/` | Other (Turkish) | Turkish | Typology test (hypothesis retracted, ISS-16) |
| `ZhouNarratives/` | Chinese (Mandarin) | Mandarin | Typology test (hypothesis retracted, ISS-16) |

Scripts read these folders directly (paths are relative to the repo root), so a fresh clone
can run the analyses without downloading anything: see the README "Setup" section.

`data/childes/not_in_use/` (superseded Spanish corpora) is kept locally only and is not
committed; no script reads it.
