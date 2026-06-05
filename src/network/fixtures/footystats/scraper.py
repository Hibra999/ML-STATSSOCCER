import json
import logging
import os
import shutil
import time
import pandas as pd
from typing import Optional
from lxml import html
from selenium.webdriver import Chrome, Firefox, Edge, ChromeOptions, FirefoxOptions, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.network.netutils import check_internet_connection


def _build_chromium_options(headless: bool) -> ChromeOptions:
    options = ChromeOptions()
    options.add_argument('--incognito')
    options.add_argument('--lang=en-US')
    if headless:
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
    return options


def _detect_brave_binary() -> str:
    for command in ('brave-browser', 'brave-browser-stable', 'brave'):
        found = shutil.which(command)
        if found:
            return found

    for path in (
            '/usr/bin/brave-browser',
            '/usr/bin/brave-browser-stable',
            '/snap/bin/brave',
            '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
            'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
            'C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
    ):
        if os.path.exists(path):
            return path
    return ''


def _resolve_brave_binary(browser_cfg: dict) -> str:
    configured = str(browser_cfg.get('brave_binary') or '').strip()
    if configured:
        return configured
    env_binary = os.environ.get('BRAVE_BINARY', '').strip()
    if env_binary:
        return env_binary
    return _detect_brave_binary()


class FootyStatsScraper:
    """Scraper de FootyStats para abrir la pagina de fixtures y leer la tabla."""

    def __init__(self, headless: Optional[bool] = None):
        self._page_load_timeout = 5.0
        self._poll_frequency = 0.5

        with open('storage/network/browser.json', mode='r') as jsonfile:
            browser_cfg = json.load(jsonfile)

        browser = str(browser_cfg.get('application', 'chrome')).lower()
        if headless is None:
            headless = browser_cfg.get('headless', True)

        if browser == 'chrome':
            self._web_driver = Chrome(options=_build_chromium_options(headless=headless))
        elif browser == 'brave':
            options = _build_chromium_options(headless=headless)
            brave_binary = _resolve_brave_binary(browser_cfg)
            if not brave_binary:
                raise RuntimeError(
                    'No se encontro el ejecutable de Brave. Configura "brave_binary" '
                    'en storage/network/browser.json o define la variable BRAVE_BINARY.'
                )
            options.binary_location = brave_binary
            self._web_driver = Chrome(options=options)
        elif browser == 'firefox':
            options = FirefoxOptions()
            options.add_argument('--incognito')
            options.set_preference('intl.accept_languages', 'en-US, en')
            if headless:
                options.add_argument('--headless')
            self._web_driver = Firefox(options=options)
        elif browser == 'edge':
            options = EdgeOptions()
            options.add_argument('--incognito')
            options.add_argument('--lang=en-US')
            if headless:
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
            self._web_driver = Edge(options=options)
        else:
            raise NotImplementedError(
                f'Navegador no implementado: "{browser}". '
                f'Usa chrome, firefox, edge o brave.'
            )

    def load_page(self, fixture_url: str) -> bool:
        """Carga FootyStats y espera la tabla de partidos."""

        # Check internet connection first.
        if not check_internet_connection():
            return False

        # Load webpage using the web driver.
        self._web_driver.get(url=fixture_url)
        self._web_driver.refresh()

        try:
            WebDriverWait(self._web_driver, timeout=self._page_load_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.full-matches-table'))
            )
        except Exception as _:
            logging.info('Se agoto el tiempo de espera para cargar la tabla de partidos.')
            return False

        time.sleep(1.0)
        return True

    def parse_fixture_table(self, date_str: str) -> Optional[pd.DataFrame]:
        """Lee la tabla de partidos ya cargada en la pagina."""

        # Reads the odd from the provided span.
        def get_odd(span) -> str:
            text = span.text

            # If text is inside child element, it attempts to read the text inside the child element instead.
            # If no odd text is found, it returns 1.0, which id the default odd value.
            if text is None:
                for child in span:
                    text = child.text

                    if text is not None:
                        break

            return text.replace('\n', '').replace('\t', '') if text is not None else '1.0'

        tree = html.fromstring(self._web_driver.page_source)
        table_elements = tree.xpath('//div[contains(@class, "full-matches-table mt1e")]')

        if len(table_elements) == 0:
            raise RuntimeError('No se encontro la tabla "full-matches-table mt1e".')

        # Searching the requested table by date.
        formatted_date_str = f'{date_str} ~'
        requested_table = None
        for table in table_elements:
            date_element = table.find('h2')

            if date_element is None:
                continue

            # if date_element.text == formatted_date_str:
            if date_element.text_content().strip() == formatted_date_str:
                requested_table = table
                break

        if requested_table is None:
            logging.info(f'No se encontro la fecha seleccionada: "{date_str}" en la tabla.')
            return None

        # Parsing fixture table.
        home_teams = []
        away_teams = []
        odds_1 = []
        odds_x = []
        odds_2 = []
        for ul in requested_table.findall('.//ul')[1:]:
            # Parsing teams.
            home_teams.append(ul.findall('.//a')[0].find('.//span').text)
            away_teams.append(ul.findall('.//a')[2].find('.//span').text)

            # Parsing odds.
            odd_spans = ul.findall('li')[-1].xpath('.//span[contains(@class, "hover-modal-parent")]')
            odd_1 = get_odd(span=odd_spans[0])
            odds_1.append(odd_1)
            odd_x = get_odd(span=odd_spans[1])
            odds_x.append(odd_x)
            odd_2 = get_odd(span=odd_spans[2])
            odds_2.append(odd_2)

        # Add year to dates.
        df = pd.DataFrame({
            'Home': home_teams,
            'Away': away_teams,
            '1': odds_1,
            'X': odds_x,
            '2': odds_2
        })
        return df

    def quit(self):
        self._web_driver.quit()
