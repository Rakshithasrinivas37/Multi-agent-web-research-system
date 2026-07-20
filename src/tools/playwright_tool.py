"""Playwright webpage rendering tool."""

from typing import Any

import httpx

from src.tools.text_utils import clean_text, title_from_html


async def render_with_playwright(url: str) -> dict[str, Any]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return await fetch_with_httpx(url, "playwright package is not installed; used httpx fallback.")

    errors = []
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page(user_agent="Mozilla/5.0")
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception as error:
                    errors.append(f"Network idle wait skipped: {error}")

                title = await page.title()
                content = await extract_visible_content(page, errors)
                table_text = await extract_table_text(page)
                if table_text:
                    content = f"{content}\n\nExtracted tables:\n{table_text}"
                final_url = page.url
                status_code = response.status if response else None
            finally:
                await browser.close()

        if status_code and status_code >= 400:
            errors.append(f"HTTP {status_code}")
        return {
            "url": final_url,
            "title": title,
            "content": content,
            "method": "playwright",
            "errors": errors,
        }
    except Exception as error:
        return await fetch_with_httpx(url, f"Playwright failed: {error}")


async def extract_visible_content(page: Any, errors: list[str]) -> str:
    chunks = []
    await scroll_page(page, errors)
    chunks.append(await body_text(page))

    await expand_interactive_content(page, errors)
    await scroll_page(page, errors)
    chunks.append(await body_text(page))

    for selector in tab_selectors():
        chunks.extend(await click_and_collect(page, selector, errors, max_clicks=8))

    return "\n\n".join(unique_text_chunks(chunks))


async def scroll_page(page: Any, errors: list[str]) -> None:
    try:
        await page.evaluate(
            """async () => {
                const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const steps = 5;
                for (let index = 1; index <= steps; index += 1) {
                    window.scrollTo(0, document.body.scrollHeight * index / steps);
                    await delay(250);
                }
                window.scrollTo(0, 0);
                await delay(250);
            }"""
        )
    except Exception as error:
        errors.append(f"Page scroll skipped: {error}")


async def expand_interactive_content(page: Any, errors: list[str]) -> None:
    selectors = [
        "details:not([open]) > summary",
        "[aria-expanded='false']",
        "[role='button'][aria-controls]",
        "button:has-text('Show more')",
        "button:has-text('See more')",
        "button:has-text('View more')",
        "button:has-text('Load more')",
        "button:has-text('Expand')",
    ]
    for selector in selectors:
        await click_elements(page, selector, errors, max_clicks=10)


def tab_selectors() -> list[str]:
    return [
        "[role='tab']",
        "button[role='tab']",
        "button[aria-selected]",
        "button[aria-controls]",
    ]


async def click_and_collect(page: Any, selector: str, errors: list[str], max_clicks: int) -> list[str]:
    chunks = []
    elements = page.locator(selector)
    try:
        count = min(await elements.count(), max_clicks)
    except Exception as error:
        errors.append(f"Could not inspect selector {selector!r}: {error}")
        return chunks

    for index in range(count):
        element = elements.nth(index)
        if not await safe_click(element):
            continue
        try:
            await page.wait_for_timeout(350)
            chunks.append(await body_text(page))
        except Exception as error:
            errors.append(f"Could not collect text after clicking {selector!r}: {error}")
    return chunks


async def click_elements(page: Any, selector: str, errors: list[str], max_clicks: int) -> None:
    elements = page.locator(selector)
    try:
        count = min(await elements.count(), max_clicks)
    except Exception as error:
        errors.append(f"Could not inspect selector {selector!r}: {error}")
        return

    for index in range(count):
        element = elements.nth(index)
        clicked = await safe_click(element)
        if clicked:
            await page.wait_for_timeout(250)


async def safe_click(element: Any) -> bool:
    try:
        if not await element.is_visible(timeout=500):
            return False
        if not await element.is_enabled(timeout=500):
            return False
        tag_name = (await element.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        href = await element.get_attribute("href")
        if tag_name == "a" and href:
            return False
        await element.click(timeout=1500)
        return True
    except Exception:
        return False


async def body_text(page: Any) -> str:
    return clean_text(await page.locator("body").inner_text(timeout=10000))


def unique_text_chunks(chunks: list[str]) -> list[str]:
    unique = []
    seen = set()
    for chunk in chunks:
        chunk = clean_text(chunk)
        key = chunk[:500]
        if chunk and key not in seen:
            unique.append(chunk)
            seen.add(key)
    return unique


async def extract_table_text(page: Any) -> str:
    rows = await page.evaluate(
        """() => Array.from(document.querySelectorAll('tr')).map((row) =>
            Array.from(row.querySelectorAll('th,td'))
                .map((cell) => cell.innerText.trim().replace(/\\s+/g, ' '))
                .filter(Boolean)
                .join(' | ')
        ).filter(Boolean)"""
    )
    unique_rows = []
    for row in rows:
        row = clean_text(row)
        if row and row not in unique_rows:
            unique_rows.append(row)
    return "\n".join(unique_rows)


async def fetch_with_httpx(url: str, warning: str = "") -> dict[str, Any]:
    errors = [warning] if warning else []
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        response = await client.get(url)

    if response.status_code >= 400:
        errors.append(f"HTTP {response.status_code}")
    return {
        "url": str(response.url),
        "title": title_from_html(response.text),
        "content": response.text,
        "method": "httpx",
        "errors": errors,
    }
