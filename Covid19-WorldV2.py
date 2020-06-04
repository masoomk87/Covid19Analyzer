import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DateTime as dt

# Country Selector
url = 'https://covid.ourworldindata.org/data/owid-covid-data.xlsx?raw=True'
country_name = 'Germany'
df_main = pd.read_excel(url, sheet_name='Sheet1')
df = df_main[df_main["location"] == country_name]

days = np.linspace(0, len(df), len(df)).astype(int)
date = df['date']

day = []
for dates in date:
    dayT = dates[5:7] + '-' + dates[8:10]
    day.append(dayT)

# 1 Daily new cases
new_cases = df['new_cases']
case_100 = np.argmax(new_cases > 100)
new_cases_roll = df.new_cases.rolling(5).mean()
y_pos = range(len(day))
plt.figure(1)
plt.title('Daily new cases' + ' - ' + country_name)
plt.bar(day[case_100:], new_cases[case_100:])
# Rotation of the bars names
plt.plot(day[case_100:], new_cases_roll[case_100:], 'r', linewidth=3)
plt.xlabel('Date')
plt.ylabel('Number of cases / day')
plt.xticks(day[case_100::2], rotation=90)
leg1 = '5 day moving average - ' + country_name
leg2 = 'Number of cases / day - ' + country_name
plt.legend([leg1, leg2])

# 2 Cumulative cases on semilog plot
cum_cases = df.new_cases.cumsum()
plt.figure(2)
plt.suptitle('Daily new cases' + ' - ' + country_name)
plt.subplot(2, 1, 1)
plt.grid(True, which="both")
plt.semilogy(day[case_100:], cum_cases[case_100:])
plt.ylim([100, max(cum_cases) + 1000])
plt.xlabel('Date')
plt.ylabel('Cumulative cases (Semilog plot)')
plt.xticks(day[case_100::2], rotation=90)
plt.subplot(2, 1, 2)
plt.grid(True, which="both")
plt.plot(day[case_100:], cum_cases[case_100:])
plt.ylim([100, max(cum_cases) + 1000])
plt.xlabel('Date')
plt.ylabel('Cumulative cases')
plt.xticks(day[case_100::2], rotation=90)

# 3 Deaths (New Deaths)
new_deaths = df['new_deaths']
new_deaths_roll = df.new_deaths.rolling(5).mean()
plt.figure(3)
plt.grid(True, which="both")
plt.title('New deaths' + ' - ' + country_name)
plt.bar(day[case_100:], new_deaths[case_100:])
plt.plot(day[case_100:], new_deaths_roll[case_100:], 'r', linewidth=3)
plt.xlabel('Date')
plt.ylabel('Number of deaths / day')
leg1 = '5 day moving average - '
leg2 = 'Number of deaths / day - '
plt.legend([leg1, leg2])
plt.xticks(day[case_100::2], rotation=90)

# 4 Death cumulative
cum_deaths = df.new_deaths.cumsum()
plt.figure(4)
plt.suptitle('Cumulative Deaths' + ' - ' + country_name)
plt.subplot(2, 1, 1)
plt.grid(True, which="both")
plt.semilogy(day[case_100:], cum_deaths[case_100:])
plt.ylim([0.1, max(cum_deaths) + (0.01 * max(cum_deaths))])
plt.xlabel('Date')
plt.ylabel('Cumulative deaths (Semilog plot)')
plt.xticks(day[case_100::2], rotation=90)
plt.subplot(2, 1, 2)
plt.grid(True, which="both")
plt.plot(day[case_100:], cum_deaths[case_100:])
plt.ylim([0.1, max(cum_deaths) + (0.01 * max(cum_deaths))])
plt.xlabel('Date')
plt.ylabel('Cumulative deaths')
plt.xticks(day[case_100::2], rotation=90)

# 5 Case doubling rate
cdr_interval = 7
cum_cases = cum_cases.to_numpy()
idx_probe = []
cdr_pts = []
cdr_idx = np.argmax(cum_cases > 100) + cdr_interval  # n+7 idx after 100th case (n=100th case)

if cum_cases[-1] > 2000:
    for cdr in range(cdr_idx, len(cum_cases)):
        cdr_pts.append(cdr_interval / (np.log2(cum_cases[cdr]) - np.log2(cum_cases[cdr - cdr_interval])))
    plt.figure(5)
    plt.title('Case Doubling Rate ' + ' ' + country_name)
    plt.plot(day[cdr_idx:], cdr_pts[:], 'o-')
    plt.xlabel('Date')
    plt.ylabel('Case Doubling Rate')
    plt.xticks(day[cdr_idx::2], rotation=90)

# Death doubling rate
ddr_interval = 7
cum_deaths = cum_deaths.to_numpy()
idx_probe = []
ddr_pts = []
ddr_idx = np.argmax(cum_deaths > 50) + cdr_interval  # n+7 idx after 100th case (n=100th case)

if cum_deaths[-1] > 50:
    for ddr in range(ddr_idx, len(cum_deaths)):
        ddr_pts.append(ddr_interval / (np.log2(cum_deaths[ddr]) - np.log2(cum_deaths[ddr - ddr_interval])))
    plt.figure(7)
    plt.title('Death Doubling Rate ' + ' ' + country_name)
    plt.plot(day[ddr_idx::2], ddr_pts[::2], 'o-')
    plt.xlabel('Date')
    plt.ylabel('Death Doubling Rate')
    plt.xticks(rotation=90)

# 7 Testing Stats
df.total_tests.fillna(0, inplace=True)
cum_tests = df.total_tests
cum_tests = cum_tests.to_numpy()
if np.sum(cum_tests) > 0:
    idx_test = cum_tests > 0
    plt.figure(8)
    plt.suptitle('Test data' + ' - ' + country_name)
    plt.subplot(3, 1, 1)
    plt.grid()
    plt.plot(days[idx_test], cum_tests[idx_test])
    plt.xlabel('Days since 31 Dec 2019')
    plt.ylabel('Total Samples tested')
    plt.subplot(3, 1, 2)
    pos_tests = cum_tests[idx_test]
    pos_samples = cum_cases[idx_test]
    pos_tests = np.append(pos_tests[0], np.diff(pos_tests))
    pos_samples = np.append(pos_samples[0], np.diff(pos_samples))
    plt.bar(days[idx_test], pos_tests)
    # plt.semilogy(days[idx_test], cum_tests[idx_test])
    plt.xlabel('Days since 31 Dec 2019')
    plt.ylabel('Daily samples tested')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.grid()
    pos_rat = pos_samples * 100 / pos_tests
    plt.plot(days[idx_test], pos_rat, 'o-')
    plt.xlabel('Days since 31 Dec 2019')
    plt.ylabel('% Positive')
    plt.ylim([0, max(pos_rat)])
    plt.subplots_adjust(hspace=0.25)

# 9 Case Fatality Rate
plt.figure(9)
plt.title('Case fatality rate' + ' - ' + country_name)
plt.plot(day[case_100:], df.total_deaths[case_100:] * 100 / df.total_cases[case_100:])
plt.xlabel('Date')
plt.ylabel('% of people confirmed dead')
plt.xticks(day[cdr_idx::2], rotation=90)

# 10 Case Growth Rate
new_cases = df.new_cases[case_100:]
count = 1
cgr = []
for cases in new_cases:
    if count == 1:
        cgr.append(0)
        old_cases = cases
        print(old_cases)
        count = count + 1
    else:
        if old_cases is 0:
            cgr.append(np.nan)
            old_cases = cases
        else:
            cgr.append(cases / old_cases)
            old_cases = cases
plt.figure(10)
plt.plot(day[case_100:], cgr)
plt.xticks(day[case_100:], rotation=90)
plt.xlabel('Date')
plt.ylabel('Case Growth Ratio')
plt.title('Case Growth Rate' + ' ' + country_name)

# 11 Death Growth Rate
new_deaths = df.new_deaths[ddr_idx:]
count = 1
dgr = []
for deaths in new_deaths:
    if count == 1:
        dgr.append(0)
        old_deaths = deaths
        count = count + 1
    else:
        if old_deaths is 0:
            dgr.append(np.nan)
            old_deaths = deaths
        else:
            dgr.append(deaths/old_deaths)
            old_deaths = deaths

plt.figure(11)
plt.plot(day[ddr_idx:], dgr)
plt.xticks(day[ddr_idx:], rotation=90)
plt.xlabel('Date')
plt.ylabel('Death Growth Ratio')
plt.title('Death Growth Rate' + ' ' + country_name)

# To include -> R0 calculation
#            -> Active Cases vs Total Cases
#            -> % Recovered vs % Dead in cases with outcome
#            -> Case Growth Ratio vs Testing Growth Ratio
#            ->
