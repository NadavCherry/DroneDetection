| pipeline | fps (5070) | total ms | stage breakdown |
|---|---|---|---|
| rt-a-classic | 36.1 | 26.0 | motion 21.1, stabilize+warp 4.4, tracker 0.5 |
| rt-b-verify256 | 24.36 | 39.2 | crop build 0.4, expert (amortized) 0.7, motion 20.9, stabilize+warp 4.4, tracker 0.5, verifier NN 12.3 |
| rt-c-full1280 | 78.07 | 10.4 | detector NN 5.9, stabilize+warp 4.5, tracker 0.1 |
| rt-d-full640 | 103.59 | 7.4 | detector NN 3.1, stabilize+warp 4.2, tracker 0.1 |
| rt-e-decimated | 28.53 | 33.2 | crop build 0.2, expert (amortized) 0.6, motion 21.2, stabilize+warp 4.5, tracker 0.5, verifier NN 6.2 |
| rt-f-single1280 | 69.89 | 12.6 | detector NN 7.8, stabilize+warp 4.4, tracker 0.3 |
