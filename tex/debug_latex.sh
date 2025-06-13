pdflatex -output-directory=./build main.tex
rm -rf biber --cache
biber ./build/main
pdflatex -output-directory=./build main.tex
pdflatex -output-directory=./build main.tex
