tesseract='C:/Program Files/Tesseract-OCR/tesseract.exe'
# Uncomment this line if, you’re rerunning the script
#rm eng.pffmtable  eng.shapetable eng.traineddata eng.unicharset unicharset font_properties eng.inttemp eng.normproto *.tr *.txt
"$tesseract" eng.bmmini.exp0.png eng.bmmini.exp0 nobatch box.train
unicharset_extractor eng.bmmini.exp0.box
echo “bmmini 0 0 0 0 0” > font_properties # tell Tesseract informations about the font
mftraining –F font_properties –U unicharset –O eng.unicharset eng.bmmini.exp0.tr
#cntraining `wrap $N “eng.bmmini.exp” “.tr”`
# rename all files created by mftraing en cntraining, add the prefix eng.:
    #mv inttemp eng.inttemp
    #mv normproto eng.normproto
    #mv pffmtable eng.pffmtable
    #mv shapetable eng.shapetable
#combine_tessdata eng.