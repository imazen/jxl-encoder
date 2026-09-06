# qf-seed lift A/B

## 1. Delivered vs requested distance (does the lift displace the target?)

`delivered_ratio` = butteraugli / requested d. 1.0 = promise kept, <1 = finer than asked. `displacement` = off_ratio / on_ratio at the same d.

| image | class | e | gate scale | d | on ratio | off ratio | displacement |
|---|---|--:|--:|--:|--:|--:|--:|
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 2 | 0.90 | 0.90 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 3 | 0.98 | 0.98 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 3.4 | 0.90 | 0.90 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 3.6 | 1.01 | 1.01 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 4 | 0.84 | 0.84 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 5 | 0.85 | 0.85 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 6 | 0.75 | 0.75 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 8 | 0.86 | 0.86 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 2 | 0.90 | 0.90 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 3 | 0.87 | 0.87 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 3.4 | 1.04 | 1.04 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 3.6 | 0.97 | 0.97 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 4 | 0.75 | 0.75 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 5 | 0.85 | 0.85 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 6 | 0.72 | 0.72 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 8 | 0.84 | 0.84 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 2 | 1.19 | 1.19 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 3 | 0.90 | 0.90 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 3.4 | 1.08 | 1.08 | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 3.6 | 0.73 | 0.98 | 1.34 ** |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 4 | 0.83 | 1.23 | 1.48 ** |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 5 | 0.85 | 1.09 | 1.29 ** |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 6 | 0.65 | 0.91 | 1.41 ** |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 8 | 0.81 | 0.85 | 1.05 |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 2 | 1.27 | 1.27 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 3 | 1.10 | 1.10 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 3.4 | 1.01 | 1.01 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 3.6 | 0.68 | 1.00 | 1.46 ** |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 4 | 0.67 | 1.13 | 1.69 ** |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 5 | 0.62 | 0.89 | 1.43 ** |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 6 | 0.57 | 1.02 | 1.79 ** |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 8 | 0.51 | 0.93 | 1.83 ** |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 2 | 1.27 | 1.27 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 3 | 1.10 | 1.10 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 3.4 | 1.00 | 1.00 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 3.6 | 0.49 | 1.01 | 2.05 ** |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 4 | 0.51 | 1.12 | 2.22 ** |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 5 | 0.48 | 0.89 | 1.84 ** |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 6 | 0.42 | 1.02 | 2.40 ** |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 8 | 0.41 | 0.93 | 2.28 ** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 2 | 1.15 | 1.15 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 3 | 1.09 | 1.09 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 3.4 | 1.07 | 1.07 | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 3.6 | 0.60 | 1.02 | 1.70 ** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 4 | 0.47 | 1.02 | 2.16 ** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 5 | 0.47 | 1.11 | 2.33 ** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 6 | 0.49 | 1.02 | 2.07 ** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 8 | 0.54 | 1.06 | 1.97 ** |
| codec_wiki | screenshot-true | 5 | 2 | 2 | 0.87 | 0.87 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 3 | 0.83 | 0.83 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 3.4 | 0.85 | 0.85 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 3.6 | 0.73 | 0.73 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 4 | 0.87 | 0.87 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 5 | 0.72 | 0.72 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 6 | 0.69 | 0.69 | 1.00 |
| codec_wiki | screenshot-true | 5 | 2 | 8 | 0.59 | 0.59 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 2 | 0.83 | 0.83 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 3 | 0.83 | 0.83 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 3.4 | 0.81 | 0.81 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 3.6 | 0.70 | 0.70 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 4 | 0.77 | 0.77 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 5 | 0.73 | 0.73 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 6 | 0.69 | 0.69 | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 8 | 0.59 | 0.59 | 1.00 |
| codec_wiki | screenshot-true | 8 | 4 | 2 | 1.31 | 1.31 | 1.00 |
| codec_wiki | screenshot-true | 8 | 4 | 3 | 0.95 | 0.95 | 1.00 |
| codec_wiki | screenshot-true | 8 | 4 | 3.4 | 0.94 | 0.94 | 1.00 |
| codec_wiki | screenshot-true | 8 | 4 | 3.6 | 0.50 | 1.00 | 2.03 ** |
| codec_wiki | screenshot-true | 8 | 4 | 4 | 0.56 | 0.84 | 1.48 ** |
| codec_wiki | screenshot-true | 8 | 4 | 5 | 0.57 | 0.78 | 1.37 ** |
| codec_wiki | screenshot-true | 8 | 4 | 6 | 0.53 | 0.96 | 1.80 ** |
| codec_wiki | screenshot-true | 8 | 4 | 8 | 0.44 | 0.74 | 1.68 ** |
| terminal | screenshot-true | 5 | 2 | 2 | 0.62 | 0.62 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 3 | 0.63 | 0.63 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 3.4 | 0.64 | 0.64 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 3.6 | 0.63 | 0.63 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 4 | 0.60 | 0.60 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 5 | 0.54 | 0.54 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 6 | 0.53 | 0.53 | 1.00 |
| terminal | screenshot-true | 5 | 2 | 8 | 0.59 | 0.59 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 2 | 0.62 | 0.62 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 3 | 0.64 | 0.64 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 3.4 | 0.64 | 0.64 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 3.6 | 0.69 | 0.69 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 4 | 0.81 | 0.81 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 5 | 0.54 | 0.54 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 6 | 0.53 | 0.53 | 1.00 |
| terminal | screenshot-true | 7 | 3 | 8 | 0.59 | 0.59 | 1.00 |
| terminal | screenshot-true | 8 | 4 | 2 | 0.47 | 0.72 | 1.54 ** |
| terminal | screenshot-true | 8 | 4 | 3 | 0.53 | 0.91 | 1.72 ** |
| terminal | screenshot-true | 8 | 4 | 3.4 | 0.51 | 0.81 | 1.60 ** |
| terminal | screenshot-true | 8 | 4 | 3.6 | 0.35 | 0.76 | 2.14 ** |
| terminal | screenshot-true | 8 | 4 | 4 | 0.52 | 0.87 | 1.68 ** |
| terminal | screenshot-true | 8 | 4 | 5 | 0.28 | 0.68 | 2.49 ** |
| terminal | screenshot-true | 8 | 4 | 6 | 0.53 | 0.68 | 1.29 ** |
| terminal | screenshot-true | 8 | 4 | 8 | 0.38 | 0.70 | 1.87 ** |

### displacement where the lift fires, vs the gate's own scale

| effort | gate scale | n firing cells | median displacement |
|--:|--:|--:|--:|
| 5 | 2 | 5 | 1.69 |
| 7 | 3 | 5 | 2.22 |
| 8 | 4 | 22 | 1.70 |

## 2. Does the lift ever pay on the RD curve?

For each lifted point, the cheapest MEASURED unlifted setting that reached the same or better butteraugli. `cost` < 1.00 means the lift genuinely won at matched quality.

| image | class | e | d | lifted bytes | its bfly | unlifted bytes @ same quality | cost |
|---|---|--:|--:|--:|--:|--:|--:|
| 7026_plots_line-00081-s684 | plots | 5 | 2 | 62441 | 1.80 | 62441 (d=2) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 3 | 52941 | 2.93 | 52941 (d=3) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 3.4 | 49877 | 3.04 | 49877 (d=3.4) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 3.6 | 48542 | 3.64 | 45998 (d=4) | 1.06 |
| 7026_plots_line-00081-s684 | plots | 5 | 4 | 45998 | 3.36 | 45998 (d=4) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 5 | 40247 | 4.23 | 40247 (d=5) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 6 | 36550 | 4.49 | 36550 (d=6) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 5 | 8 | 30663 | 6.86 | 30663 (d=8) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 2 | 63332 | 1.80 | 63332 (d=2) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3 | 53851 | 2.60 | 53851 (d=3) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 3.4 | 50767 | 3.54 | 46996 (d=4) | 1.08 |
| 7026_plots_line-00081-s684 | plots | 7 | 3.6 | 49461 | 3.49 | 46996 (d=4) | 1.05 |
| 7026_plots_line-00081-s684 | plots | 7 | 4 | 46996 | 2.98 | 46996 (d=4) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 5 | 40614 | 4.25 | 40614 (d=5) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 6 | 36917 | 4.33 | 36917 (d=6) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 7 | 8 | 31052 | 6.69 | 31052 (d=8) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 2 | 56489 | 2.39 | 56489 (d=2) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 3 | 47096 | 2.69 | 47096 (d=3) | 1.00 |
| 7026_plots_line-00081-s684 | plots | 8 | 3.4 | 44237 | 3.66 | 42568 (d=3.6) | 1.04 |
| 7026_plots_line-00081-s684 | plots | 8 | 3.6 | 64262 | 2.64 | 50472 (d=2.5) | 1.27 |
| 7026_plots_line-00081-s684 | plots | 8 | 4 | 70143 | 3.31 | 47096 (d=3) | 1.49 |
| 7026_plots_line-00081-s684 | plots | 8 | 5 | 61365 | 4.24 | 42568 (d=3.6) | 1.44 |
| 7026_plots_line-00081-s684 | plots | 8 | 6 | 48081 | 3.90 | 42568 (d=3.6) | 1.13 |
| 7026_plots_line-00081-s684 | plots | 8 | 8 | 40045 | 6.45 | 31165 (d=6) | 1.28 |
| 9291_gen_products-beauty_b | ai-products | 5 | 2 | 50773 | 2.54 | 50773 (d=2) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 3 | 36670 | 3.30 | 36670 (d=3) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 3.4 | 32907 | 3.45 | 32907 (d=3.4) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 5 | 3.6 | 51267 | 2.46 | 63745 (d=1.5) | 0.80 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 5 | 4 | 46901 | 2.66 | 50773 (d=2) | 0.92 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 5 | 5 | 38496 | 3.10 | 42565 (d=2.5) | 0.90 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 5 | 6 | 33128 | 3.40 | 36670 (d=3) | 0.90 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 5 | 8 | 26081 | 4.08 | 31424 (d=3.6) | 0.83 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 7 | 2 | 50775 | 2.54 | 50775 (d=2) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3 | 36378 | 3.30 | 36378 (d=3) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3.4 | 32732 | 3.40 | 32732 (d=3.4) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 7 | 3.6 | 67618 | 1.77 | 86860 (d=1) | 0.78 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 7 | 4 | 62238 | 2.03 | 63747 (d=1.5) | 0.98 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 7 | 5 | 51951 | 2.41 | 63747 (d=1.5) | 0.81 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 7 | 6 | 44974 | 2.54 | 50775 (d=2) | 0.89 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 7 | 8 | 34822 | 3.28 | 41928 (d=2.5) | 0.83 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 8 | 2 | 45780 | 2.30 | 45780 (d=2) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 3 | 31767 | 3.28 | 31767 (d=3) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 3.4 | 28355 | 3.64 | 28355 (d=3.4) | 1.00 |
| 9291_gen_products-beauty_b | ai-products | 8 | 3.6 | 56720 | 2.15 | 59214 (d=1.5) | 0.96 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 8 | 4 | 62431 | 1.88 | 69985 (d=1.25) | 0.89 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 8 | 5 | 51629 | 2.37 | 45780 (d=2) | 1.13 |
| 9291_gen_products-beauty_b | ai-products | 8 | 6 | 35763 | 2.97 | 37202 (d=2.5) | 0.96 **WIN** |
| 9291_gen_products-beauty_b | ai-products | 8 | 8 | 26928 | 4.30 | 24939 (d=4) | 1.08 |
| codec_wiki | screenshot-true | 5 | 2 | 27618 | 1.74 | 27618 (d=2) | 1.00 |
| codec_wiki | screenshot-true | 5 | 3 | 24149 | 2.49 | 24149 (d=3) | 1.00 |
| codec_wiki | screenshot-true | 5 | 3.4 | 22453 | 2.90 | 21623 (d=3.6) | 1.04 |
| codec_wiki | screenshot-true | 5 | 3.6 | 21623 | 2.64 | 21623 (d=3.6) | 1.00 |
| codec_wiki | screenshot-true | 5 | 4 | 20391 | 3.49 | 20391 (d=4) | 1.00 |
| codec_wiki | screenshot-true | 5 | 5 | 18291 | 3.61 | 18291 (d=5) | 1.00 |
| codec_wiki | screenshot-true | 5 | 6 | 16928 | 4.13 | 16928 (d=6) | 1.00 |
| codec_wiki | screenshot-true | 5 | 8 | 14167 | 4.75 | 14167 (d=8) | 1.00 |
| codec_wiki | screenshot-true | 7 | 2 | 29163 | 1.67 | 29163 (d=2) | 1.00 |
| codec_wiki | screenshot-true | 7 | 3 | 23824 | 2.50 | 23824 (d=3) | 1.00 |
| codec_wiki | screenshot-true | 7 | 3.4 | 22511 | 2.77 | 21711 (d=3.6) | 1.04 |
| codec_wiki | screenshot-true | 7 | 3.6 | 21711 | 2.51 | 21711 (d=3.6) | 1.00 |
| codec_wiki | screenshot-true | 7 | 4 | 20706 | 3.07 | 20706 (d=4) | 1.00 |
| codec_wiki | screenshot-true | 7 | 5 | 18781 | 3.67 | 18781 (d=5) | 1.00 |
| codec_wiki | screenshot-true | 7 | 6 | 16930 | 4.13 | 16930 (d=6) | 1.00 |
| codec_wiki | screenshot-true | 7 | 8 | 14167 | 4.75 | 14167 (d=8) | 1.00 |
| codec_wiki | screenshot-true | 8 | 2 | 25488 | 2.63 | 22528 (d=2.5) | 1.13 |
| codec_wiki | screenshot-true | 8 | 3 | 20387 | 2.84 | 20387 (d=3) | 1.00 |
| codec_wiki | screenshot-true | 8 | 3.4 | 19183 | 3.19 | 19183 (d=3.4) | 1.00 |
| codec_wiki | screenshot-true | 8 | 3.6 | 28260 | 1.78 | 29288 (d=1.5) | 0.96 **WIN** |
| codec_wiki | screenshot-true | 8 | 4 | 31135 | 2.26 | 29288 (d=1.5) | 1.06 |
| codec_wiki | screenshot-true | 8 | 5 | 28595 | 2.85 | 20387 (d=3) | 1.40 |
| codec_wiki | screenshot-true | 8 | 6 | 22768 | 3.20 | 19183 (d=3.4) | 1.19 |
| codec_wiki | screenshot-true | 8 | 8 | 19101 | 3.53 | 17878 (d=4) | 1.07 |
| terminal | screenshot-true | 5 | 2 | 35502 | 1.25 | 35502 (d=2) | 1.00 |
| terminal | screenshot-true | 5 | 3 | 29894 | 1.89 | 29894 (d=3) | 1.00 |
| terminal | screenshot-true | 5 | 3.4 | 28820 | 2.18 | 28820 (d=3.4) | 1.00 |
| terminal | screenshot-true | 5 | 3.6 | 27847 | 2.27 | 27847 (d=3.6) | 1.00 |
| terminal | screenshot-true | 5 | 4 | 26311 | 2.39 | 26311 (d=4) | 1.00 |
| terminal | screenshot-true | 5 | 5 | 24364 | 2.72 | 24364 (d=5) | 1.00 |
| terminal | screenshot-true | 5 | 6 | 21654 | 3.20 | 21654 (d=6) | 1.00 |
| terminal | screenshot-true | 5 | 8 | 19974 | 4.73 | 19974 (d=8) | 1.00 |
| terminal | screenshot-true | 7 | 2 | 35518 | 1.25 | 35518 (d=2) | 1.00 |
| terminal | screenshot-true | 7 | 3 | 30187 | 1.92 | 30187 (d=3) | 1.00 |
| terminal | screenshot-true | 7 | 3.4 | 29064 | 2.17 | 29064 (d=3.4) | 1.00 |
| terminal | screenshot-true | 7 | 3.6 | 28322 | 2.49 | 28322 (d=3.6) | 1.00 |
| terminal | screenshot-true | 7 | 4 | 27203 | 3.22 | 21689 (d=6) | 1.25 |
| terminal | screenshot-true | 7 | 5 | 24356 | 2.72 | 24356 (d=5) | 1.00 |
| terminal | screenshot-true | 7 | 6 | 21689 | 3.20 | 21689 (d=6) | 1.00 |
| terminal | screenshot-true | 7 | 8 | 19992 | 4.73 | 19992 (d=8) | 1.00 |
| terminal | screenshot-true | 8 | 2 | 39548 | 0.93 | 38149 (d=1) | 1.04 |
| terminal | screenshot-true | 8 | 3 | 33312 | 1.60 | 29681 (d=2) | 1.12 |
| terminal | screenshot-true | 8 | 3.4 | 32890 | 1.73 | 29681 (d=2) | 1.11 |
| terminal | screenshot-true | 8 | 3.6 | 30977 | 1.27 | 33063 (d=1.5) | 0.94 **WIN** |
| terminal | screenshot-true | 8 | 4 | 34345 | 2.06 | 29681 (d=2) | 1.16 |
| terminal | screenshot-true | 8 | 5 | 30314 | 1.37 | 33063 (d=1.5) | 0.92 **WIN** |
| terminal | screenshot-true | 8 | 6 | 24884 | 3.17 | 23796 (d=3.6) | 1.05 |
| terminal | screenshot-true | 8 | 8 | 21039 | 3.00 | 23796 (d=3.6) | 0.88 **WIN** |

wins (lift cheaper at matched quality): **17**; losses: 24; unbracketed: 0

### all cells (non-firing ones score 1.00 by construction)

| class | cells | median cost | best (min) | worst (max) |
|---|--:|--:|--:|--:|
| ai-products | 24 | 0.98 | 0.78 | 1.13 |
| plots | 24 | 1.00 | 1.00 | 1.49 |
| screenshot-true | 48 | 1.00 | 0.88 | 1.40 |

### FIRING cells only — the honest answer to "does the lift ever pay?"

| class | firing cells | median cost | wins <0.98 | losses >1.02 | best | worst |
|---|--:|--:|--:|--:|--:|--:|
| ai-products | 15 | 0.90 | 13 | 2 | 0.78 | 1.13 |
| plots | 5 | 1.28 | 0 | 5 | 1.13 | 1.49 |
| screenshot-true | 13 | 1.06 | 4 | 9 | 0.88 | 1.40 |
| **all** | 33 | 0.98 | 17 | 16 | 0.78 | 1.49 |
