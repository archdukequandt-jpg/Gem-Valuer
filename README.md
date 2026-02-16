# Industry Standard Gem Valuer v25_mod (Streamlit)
# https://gem-valuer-i3jrcvubllekgkxshdjzpv.streamlit.app/

Wide-layout Streamlit app for gemstone identification & valuation with hydrostatic SG, 3D preview,
DB-driven pricing, regional multipliers, metals & energy equivalents, sticky order panel, and rich PDF export.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL printed by Streamlit.

## Notes
- Use the sidebar to set weight/dimensions, choose cut, and (optionally) enter hydrostatic SG.
- Toggle Retail/Wholesale presets and edit regional multipliers live.
- Metals prices can be manual or pulled from Yahoo Finance (GC=F, SI=F, PL=F).
- Energy equivalents default to manual prices; optional financial proxies used for NG/Crude.
- Generate a PDF report (requires `reportlab`).

Created by Ryan Childs. AI assistance: ChatGPT 5 Thinking.
