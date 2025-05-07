import os
import time
import pandas as pd
import random
import re
import json
import undetected_chromedriver as uc
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from urllib.parse import urlparse, parse_qs
import wikipedia
import wikipediaapi