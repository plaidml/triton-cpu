#!/usr/bin/env python

import io
import sys
import re
from pprint import pprint

EXAMPLE_RAW = """

RUN: xsmm-loop-collapse-pad-b | threads 1 | type f32 |
Compilation completed in 0.18 seconds
triton_cpu_output_with_torch.float32_inputs=tensor([[ 34.1671,  -0.6265,   2.1590,  ..., -24.7347,  45.6671,  12.1024],
        [-12.4534, -18.2695,   5.6328,  ...,   3.7349, -24.7166,  18.7390],
        [ 10.1346,   7.3148, -15.2706,  ...,  33.5417, -18.3999, -61.9606],
        ...,
        [-25.3489,  16.8383,  25.1225,  ...,  22.5323, -34.2459,  22.0000],
        [  9.3619,  47.0127,  12.2615,  ..., -32.7080, -19.9729,   7.6856],
        [-12.7560, -20.7132,  14.4755,  ...,  23.1968, -14.5273,  19.9374]])
torch_cpu_output_with_torch.float32_inputs=tensor([[ 34.1671,  -0.6265,   2.1590,  ..., -24.7347,  45.6671,  12.1024],
        [-12.4534, -18.2695,   5.6328,  ...,   3.7349, -24.7166,  18.7390],
        [ 10.1346,   7.3148, -15.2706,  ...,  33.5417, -18.3999, -61.9606],
        ...,
        [-25.3489,  16.8383,  25.1225,  ...,  22.5323, -34.2459,  22.0001],
        [  9.3619,  47.0127,  12.2615,  ..., -32.7080, -19.9729,   7.6856],
        [-12.7560, -20.7132,  14.4755,  ...,  23.1968, -14.5273,  19.9374]])
✅ TritonCPU and TorchCPU match
matmul-performance-float32 (BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=4):
         M       N       K  TritonCPU 1   TritonCPU  TorchCPU (native)  TorchCPU (compile)
0    256.0   256.0   256.0    83.403663   86.609351         109.645665           84.228830
1    384.0   384.0   384.0   118.172318  112.952268         154.620005          128.960579
2    512.0   512.0   512.0   145.178330  150.415886         172.755137          155.458527
3    640.0   640.0   640.0   158.102961  159.147336         179.373231          168.991518
4    768.0   768.0   768.0   170.901118  166.394759         189.379048          180.215272
5    896.0   896.0   896.0   173.209013  170.596176         191.698928          183.229546
6   1024.0  1024.0  1024.0   175.208011  174.800597         194.669945          186.816177
7   1152.0  1152.0  1152.0   177.620527  177.998994         196.739417          190.380273
8   1280.0  1280.0  1280.0   178.272447  177.950755         197.373697          191.638535
9   1408.0  1408.0  1408.0   177.691071  177.166407         198.105671          192.891577
10  1536.0  1536.0  1536.0   179.486495  179.378247         200.192030          195.017565
11  1664.0  1664.0  1664.0   179.248105  179.507834         200.791858          196.084638
12  1792.0  1792.0  1792.0   182.001057  181.257097         201.475120          197.201926
13  1920.0  1920.0  1920.0   181.059853  180.905985         202.465018          197.948262
14  2048.0  2048.0  2048.0   174.489858  174.177850         202.613771          197.633177
15  2176.0  2176.0  2176.0   173.737525  173.860443         202.586549          197.632979
16  2304.0  2304.0  2304.0   181.600634  180.594629         203.721915          198.211466
17  2432.0  2432.0  2432.0   172.379615  172.719040         203.159758          196.856556
18  2560.0  2560.0  2560.0   172.280768  172.281929         203.627782          197.513322
19  2688.0  2688.0  2688.0   174.945734  174.326542         203.992521          197.302048
20  2816.0  2816.0  2816.0   169.013883  168.441644         203.220474          197.478341
21  2944.0  2944.0  2944.0   162.184995  162.847548         204.051992          197.960815
22  3072.0  3072.0  3072.0   163.697817  163.423039         203.862434          198.357447
23  3200.0  3200.0  3200.0   149.726419  149.567171         203.188350          197.359947
24  3328.0  3328.0  3328.0   134.371527  135.183037         203.453161          198.094136
25  3456.0  3456.0  3456.0   135.271462  135.038314         203.551767          198.712336
26  3584.0  3584.0  3584.0   130.291836  133.235340         203.152519          198.433603
27  3712.0  3712.0  3712.0   118.380344  118.466631         203.676783          198.591036
28  3840.0  3840.0  3840.0   103.745840  104.033130         203.744578          199.196191
29  3968.0  3968.0  3968.0   101.263121  101.414464         203.518240          199.310145
30  4096.0  4096.0  4096.0   100.686834  100.550802         203.561303          199.374060

real    4m51.972s
user    4m42.483s
sys     0m8.191s


RUN: xsmm-loop-collapse-pad-b | threads 1 | type f32 | --external-pad
triton_cpu_output_with_torch.float32_inputs=tensor([[ 34.1671,  -0.6265,   2.1590,  ..., -24.7347,  45.6671,  12.1024],
        [-12.4534, -18.2695,   5.6328,  ...,   3.7349, -24.7166,  18.7390],
        [ 10.1346,   7.3148, -15.2706,  ...,  33.5417, -18.3999, -61.9606],
        ...,
        [-25.3489,  16.8383,  25.1225,  ...,  22.5323, -34.2459,  22.0000],
        [  9.3619,  47.0127,  12.2615,  ..., -32.7080, -19.9729,   7.6856],
        [-12.7560, -20.7132,  14.4755,  ...,  23.1968, -14.5273,  19.9374]])
torch_cpu_output_with_torch.float32_inputs=tensor([[ 34.1671,  -0.6265,   2.1590,  ..., -24.7347,  45.6671,  12.1024],
        [-12.4534, -18.2695,   5.6328,  ...,   3.7349, -24.7166,  18.7390],
        [ 10.1346,   7.3148, -15.2706,  ...,  33.5417, -18.3999, -61.9606],
        ...,
        [-25.3489,  16.8383,  25.1225,  ...,  22.5323, -34.2459,  22.0001],
        [  9.3619,  47.0127,  12.2615,  ..., -32.7080, -19.9729,   7.6856],
        [-12.7560, -20.7132,  14.4755,  ...,  23.1968, -14.5273,  19.9374]])
✅ TritonCPU and TorchCPU match
matmul-performance-float32 (BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=4):
         M       N       K  TritonCPU 1   TritonCPU  TorchCPU (native)  TorchCPU (compile)
0    256.0   256.0   256.0    95.471631   98.265890         112.368411           87.885404
1    384.0   384.0   384.0   129.248441  122.174338         149.817943          131.574194
2    512.0   512.0   512.0   149.178690  151.218530         169.937434          159.182795
3    640.0   640.0   640.0   167.232716  118.564673         180.703670          170.270316
4    768.0   768.0   768.0   175.846995  172.626202         189.502467          178.299835
5    896.0   896.0   896.0   176.006857  175.027577         191.556647          184.110314
6   1024.0  1024.0  1024.0   180.857766  179.685184         194.075400          186.725336
7   1152.0  1152.0  1152.0   181.496332  181.193093         196.787030          186.198787
8   1280.0  1280.0  1280.0   179.846308  180.692905         197.572326          191.009934
9   1408.0  1408.0  1408.0   182.306710  181.520470         198.470077          193.463241
10  1536.0  1536.0  1536.0   183.377090  183.402562         200.143021          194.744453
11  1664.0  1664.0  1664.0   182.666496  183.077222         200.742133          195.734441
12  1792.0  1792.0  1792.0   183.960046  183.498509         201.491361          196.924624
13  1920.0  1920.0  1920.0   183.097539  183.964776         202.125356          198.365721
14  2048.0  2048.0  2048.0   176.277142  176.468070         202.511711          197.894974
15  2176.0  2176.0  2176.0   175.120826  171.976788         202.786973          198.321492
16  2304.0  2304.0  2304.0   183.139789  182.933102         203.565949          198.095454
17  2432.0  2432.0  2432.0   175.371643  175.348531         202.917206          196.890805
18  2560.0  2560.0  2560.0   176.820832  176.390099         203.205968          197.340221
19  2688.0  2688.0  2688.0   172.339077  171.649789         203.587145          197.836050
20  2816.0  2816.0  2816.0   168.610309  168.137211         203.160750          194.059775
21  2944.0  2944.0  2944.0   159.286557  159.556802         203.800123          197.571320
22  3072.0  3072.0  3072.0   146.379124  147.117688         203.919870          197.537656
23  3200.0  3200.0  3200.0   154.014292  152.942192         203.329598          197.526776
24  3328.0  3328.0  3328.0   148.388781  148.546945         203.150722          197.837039
25  3456.0  3456.0  3456.0   136.279810  135.694116         203.733133          198.716759
26  3584.0  3584.0  3584.0   131.772059  131.862904         203.429997          198.122081
27  3712.0  3712.0  3712.0   119.918836  119.448004         203.453728          198.622173
28  3840.0  3840.0  3840.0   104.304726  104.364569         203.724757          199.117011
29  3968.0  3968.0  3968.0   101.905386  102.130741         203.432411          199.140717
30  4096.0  4096.0  4096.0   101.660556  101.606180         203.126515          199.243746

real    4m53.102s
user    4m43.192s
sys     0m8.617s"""


def convert_all(in_file):
  metadata = None
  csv_lines = []
  count_new_line_sep = 0
  for line in in_file:
    if line.startswith('sys '):
      yield (csv_lines, metadata)
      metadata = None
      csv_lines = []
    if line in {"", "\n"}:
      count_new_line_sep += 1
      if count_new_line_sep >= 2:
        yield (csv_lines, metadata)
        metadata = None
        csv_lines = []
        count_new_line_sep = 0
    if line.startswith("RUN"):
      match = re.match(r"RUN: ([\w-]+) \| threads (\d+) \| type (\w+) ?\|? ?(--external-pad)?.*", line)
      metadata = {'config': match[1], 'threads': match[2], 'type': match[3], 'line': line}
    if m := re.match(r"\d+ +(\d.*)", line): # data row
      csv_line = re.subn("[, ]+", ",", m[1])
      csv_lines += [csv_line[0]+"\n"]


if __name__ == "__main__":
  if (sys.argv[1] == "+"):
    in_file = io.StringIO(EXAMPLE_RAW)
  else:
    in_file = sys.stdin if (len(sys.argv) < 2 or sys.argv[1] == '-') else open(sys.argv[1])

  line1 = in_file.readline()
  line2 = in_file.readline()
  assert(line1 == "\n" and line2 == "\n")
  converted = convert_all(in_file)

  for csv_lines, metadata in converted:
    print(metadata['line'], end='')
    print("M,N,K,TritonCPU (1core),TritonCPU,TorchCPU (native),TorchCPU (compile)")
    for csv_line in csv_lines:
      print(csv_line, end='') # csv_line includes its own newline


