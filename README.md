This weeks todos:
- check the quality of photon data. are there significant outliers during nighttime?
    - with a ppfd threshold of 5, there are only 22 outliers between the hours of 8pm and 4am
    - ppfd should be fine to use for daytime analysis
    - visually as well, there are not a lot of outliers that *include* nighttime data
        - there are some days where ppfd is low nearly all day, so some days might get *excluded*
- try some different ANN architectures
|Layer dimensions | Avg R-squared |
| --------------- | ----- |
|  4          | 0.634 |
|  6          | 0.616 |
|  8          | 0.646 |
|  10         | 0.631 |
|  12        | 0.628 |
|  14         | 0.653 |
|  20         | 0.609 |

- start lookup up time series predictors
