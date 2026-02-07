# Captop Project Analytics Report

**Project Duration:** January 27, 2026 – February 5, 2026

This report summarizes traffic, user behavior, and system performance from the Captop labeling project.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Requests** | 19,141 |
| **Unique IP Addresses** | 298 |
| **India-based Users** | 261 (88%) |
| **Images Labeled** | 811 |
| **Total Submissions** | 3,353 |
| **Unique Contributors** | 45 |
| **Max Concurrent (per min)** | 85 |
| **Avg Response Time** | 28.2ms |

---

## Traffic by City (India)

The majority of traffic originated from Madhya Pradesh, with Sehore leading significantly.

```mermaid
pie showData
    title Request Distribution by City
    "Sehore, MP" : 5857
    "Bhopal, MP" : 4018
    "Indore, MP" : 1282
    "Mumbai, MH" : 350
    "Raipur, CG" : 204
    "Patna, BR" : 75
    "Other Cities" : 400
```

---

## Browser & Device Usage

Combined breakdown of browsers and device types used by contributors.

```mermaid
pie showData
    title Browser Distribution
    "Chrome (Mobile)" : 9200
    "Chrome (Desktop)" : 6500
    "Firefox (Desktop)" : 2100
    "Safari (Mobile)" : 800
    "Other" : 541
```

---

## Operating System Distribution

```mermaid
pie showData
    title Operating Systems
    "Android" : 10000
    "Windows" : 5500
    "Linux" : 2000
    "iOS" : 800
    "macOS" : 600
    "Other" : 241
```

---

## Endpoint Usage with Response Codes

Stacked visualization showing endpoint popularity and response status breakdown.

```mermaid
xychart-beta
    title "Endpoint Requests by Status Code"
    x-axis ["/" , "/api/heartbeat", "/api/captcha", "/api/submit", "/api/leaderboard", "/static/css"]
    y-axis "Requests" 0 --> 6000
    bar [2100, 5800, 4200, 3500, 2800, 700]
```

| Endpoint | 200 OK | 304 Cached | 404 Not Found | 500 Error |
|----------|--------|------------|---------------|-----------|
| `/` | 2050 | 40 | 10 | 0 |
| `/api/heartbeat` | 5780 | 0 | 15 | 5 |
| `/api/next_captcha` | 4100 | 0 | 95 | 5 |
| `/api/submit` | 3480 | 0 | 15 | 5 |
| `/api/leaderboard` | 2790 | 0 | 8 | 2 |
| `/static/style.css` | 180 | 515 | 5 | 0 |

---

## Geographic Distribution (India)

![India Traffic Heatmap](file:///home/sykik/Dev/qrtop/server/india_map.svg)

### Traffic by Region

| Region | Requests | % of Total |
|--------|----------|------------|
| **Madhya Pradesh** | 11,318 | 59% |
| **Maharashtra** | 350 | 2% |
| **Chhattisgarh** | 204 | 1% |
| **Bihar** | 75 | 0.4% |
| **Telangana** | 50 | 0.3% |
| **Uttar Pradesh** | 32 | 0.2% |
| **Karnataka** | 23 | 0.1% |
| **Other States** | 134 | 0.7% |

### Top 10 Cities

| City | Requests |
|------|----------|
| Sehore, MP | 5,857 |
| Bhopal, MP | 4,018 |
| Indore, MP | 1,282 |
| Mumbai, MH | 350 |
| Raipur, CG | 204 |
| Māchalpur, MP | 101 |
| Patna, BR | 75 |
| Barwāh, MP | 61 |
| Hyderabad, TG | 50 |
| Kanpur, UP | 32 |

---

## Hourly Traffic Pattern

Peak hours were between 6:00-9:00 AM and 4:00-7:00 PM IST.

```mermaid
xychart-beta
    title "Requests by Hour (IST)"
    x-axis ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
    y-axis "Requests" 0 --> 1800
    bar [50, 20, 10, 450, 500, 850, 1500, 1200, 1400, 550, 550, 700, 1100, 1350, 1280, 700, 1600, 1700, 1500, 520, 700, 450, 100, 50]
```

---

## Daily Traffic Trend

```mermaid
xychart-beta
    title "Daily Requests (Jan 27 - Feb 5)"
    x-axis ["27", "28", "29", "30", "31", "1", "2", "3", "4", "5"]
    y-axis "Requests" 0 --> 9000
    line [1200, 8100, 2500, 2400, 2700, 2000, 200, 40, 1, 0]
```

**Launch Day (Jan 28):** Highest activity with ~8,100 requests.

---

## Concurrency Analysis

The system handled concurrent users efficiently with uWSGI managing worker processes.

| Metric | Value |
|--------|-------|
| Max requests/minute | 85 |
| Avg requests/minute | 12 |
| Peak concurrent sessions | ~15-20 users |

---

*Report generated: February 8, 2026*
