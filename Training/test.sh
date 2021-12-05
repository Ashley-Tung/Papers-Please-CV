tesseract='C:/Program Files/Tesseract-OCR/tesseract.exe'
N=3 # set accordingly to the number of files that you have
for i in $(seq 1 $N); do
    "$tesseract" eng.bmmini.exp$i.png eng.bmmini.exp$i batch.nochop makebox
done