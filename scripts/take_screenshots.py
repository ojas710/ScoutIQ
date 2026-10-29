"""Capture real screenshots of the running app (requires: playwright + a Chromium build)."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"; OUT.mkdir(parents=True, exist_ok=True)

def settle(pg, s=4):
    pg.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60000)
    time.sleep(s)
    pg.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=90000)
    time.sleep(1.5)

def pick(pg, nth, text):
    box = pg.locator('[data-testid="stSelectbox"] input').nth(nth)
    box.click(); box.fill(text); time.sleep(1.2); pg.keyboard.press("Enter"); settle(pg, 3)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1400, "height": 2200})
    pg.goto(BASE); settle(pg, 8); pg.screenshot(path=OUT / "01_dashboard.png", full_page=True)
    pg.goto(f"{BASE}/player_valuation"); settle(pg, 5); pick(pg, 0, "Erling Haaland"); pg.screenshot(path=OUT / "02_player_valuation.png", full_page=True)
    pg.goto(f"{BASE}/player_comparison"); settle(pg, 5)
    ms = pg.locator('[data-testid="stMultiSelect"] input').first
    for n in ["Erling Haaland", "Kylian Mbapp", "Harry Kane"]:
        ms.click(); ms.fill(n); time.sleep(1.2); pg.keyboard.press("Enter"); settle(pg, 2)
    pg.keyboard.press("Escape"); time.sleep(1)
    pg.screenshot(path=OUT / "03_player_comparison.png", full_page=True)
    pg.goto(f"{BASE}/similar_players"); settle(pg, 5); pick(pg, 0, "Kylian Mbapp"); pg.screenshot(path=OUT / "04_similar_players.png", full_page=True)
    pg.goto(f"{BASE}/undervalued_players"); settle(pg, 6); pg.screenshot(path=OUT / "05_undervalued_players.png", full_page=True)
    pg.goto(f"{BASE}/model_performance"); settle(pg, 6); pg.screenshot(path=OUT / "06_model_performance.png", full_page=True)
    pg.goto(f"{BASE}/custom_predictor"); settle(pg, 5)
    pg.get_by_role("button", name="Predict market value").click(); settle(pg, 6)
    pg.screenshot(path=OUT / "07_custom_predictor.png", full_page=True)
    b.close()
print(sorted(x.name for x in OUT.glob("*.png")))
