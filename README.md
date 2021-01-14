# NetKI

NetKI: A Kirchhoff Index Based Statistical Graph Embedding in Nearly Linear Time <br />
required packages:<br />

numpy <br />
networkx==2.2 <br />
julia <br />
scipy.sparse <br />

to connect with julia, install the following packages <b <br />r />
install julia interpreter: https://julialang.org/downloads/ <br />
open julia and type<br />
using Pkg <br />
Pkg.add("Laplacians") <br />
Pkg.add("PyCall") <br />
Pkg.add("IJulia") <br />
Pkg.add("Distributions")<br />
Pkg.build("PyCall") <br />

type the following command to run the code: <br />
python main.py<br />

Please cite our paper with the following information;

@article{said2020netki,
  title={NetKI: A Kirchhoff Index Based Statistical Graph Embedding in Nearly Linear Time},
  author={Said, Anwar and Hassan, Saeed-Ul and Abbas, Waseem and Shabbir, Mudassir},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
