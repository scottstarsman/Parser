# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:11:18 2018

@author: SStarsman
"""

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure


def parse_layout(layout):
    """Function to recursively parse the layout tree."""
    for lt_obj in layout:
        #print(lt_obj.__class__.__name__)
        #print(lt_obj.bbox)
        if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            print(lt_obj.get_text() + str(lt_obj.bbox))
            print()
        elif isinstance(lt_obj, LTFigure):
            parse_layout(lt_obj)  # Recursive


fp = open('11-31-41-1602 Lens.pdf', 'rb')
parser = PDFParser(fp)
doc = PDFDocument(parser)

rsrcmgr = PDFResourceManager()
laparams = LAParams()
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)
for page in PDFPage.create_pages(doc):
    interpreter.process_page(page)
    layout = device.get_result()
    parse_layout(layout)