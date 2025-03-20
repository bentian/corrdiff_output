# Experiment Configuration

## Experiment Name Prefix

| Prefix | [Baseline variables](https://docs.google.com/spreadsheets/d/1K0Ngb0IIYs6aUgZuv6XRsd7Y9H8PN248V7n5w2M-EHE/edit?gid=2035957628#gid=2035957628&range=D:D) | [Terrain height](https://docs.google.com/spreadsheets/d/1K0Ngb0IIYs6aUgZuv6XRsd7Y9H8PN248V7n5w2M-EHE/edit?gid=2035957628#gid=2035957628&range=36:36) | [Weighted precipitation](https://docs.google.com/spreadsheets/d/1K0Ngb0IIYs6aUgZuv6XRsd7Y9H8PN248V7n5w2M-EHE/edit?gid=2035957628#gid=2035957628&range=39:39) | [Slope & Aspect](https://docs.google.com/spreadsheets/d/1K0Ngb0IIYs6aUgZuv6XRsd7Y9H8PN248V7n5w2M-EHE/edit?gid=2035957628#gid=2035957628&range=37:38) |
| ---    | :---:    | :---:     | :---:                  | :---:          |
| `BL`   | :white_check_mark: | :x: | :x: | :x: |
| `D1`   | :white_check_mark: | :white_check_mark: | :x: | :x: |
| `D2`   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| `D3`   | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |

## Experiment Name Suffix

| Suffix       | Training data | Training duration | Landmask?          | # ensembles | Test data |
| ---          | :---:         | :---:             | :---:              | :---:       | :---:     |
| `2M`         | 2018-2022     | 2M                | :white_check_mark: | 1           | 2023      |
| `2M_ens64`   | 2018-2022     | 2M                | :white_check_mark: | *64*        | 2023      |
| `4M`         | 2018-2022     | *4M*              | :white_check_mark: | 1           | 2023      |
| `4M_1322`    | *2013-2022*   | *4M*              | :white_check_mark: | 1           | 2023      |
| `extreme_1M` | *2018-2022 w/ extreme data* | *2M + 1M* | :white_check_mark: | 1         | 2023      |
| `nomask_2M`  | 2018-2022     | 2M                | :x:       | 1           | 2023      |
