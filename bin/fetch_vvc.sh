#! /bin/bash
#

# Script ensures VVC is built on the system

build_vvc() {
	git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vvc
	mkdir -p vvc/build
	cd vvc/build
	cmake .. -DCMAKE_BUILD_TYPE=Release
	make -j
}

[ -d "vvc" ] || build_vvc
