import asyncio
from playwright.async_api import async_playwright


async def scrape_page(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        )

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            title = await page.title()
            text = await page.locator("body").inner_text(timeout=10000)

            return {
                "url": page.url,
                "title": title,
                "content": text,
                "content_length": len(text),
                "status": "success",
            }

        except Exception as error:
            return {
                "url": url,
                "title": "",
                "content": "",
                "content_length": 0,
                "status": "failed",
                "error": str(error),
            }

        finally:
            await browser.close()


async def main():
    url = "https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735"
    result = await scrape_page(url)

    print("URL:", result["url"])
    print("Title:", result["title"])
    print("Status:", result["status"])
    print("Content length:", result["content_length"])
    print(result["content"][:3000])


if __name__ == "__main__":
    asyncio.run(main())