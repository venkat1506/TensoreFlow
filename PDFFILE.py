# importing required modules
import PyPDF2

# creating a pdf file object
pdfFileObj = open('Invoice OD114526196461874000.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

print(pdfReader.numPages)

pageObj = pdfReader.getPage(0)

# extracting text from page
print(pageObj.extractText())
# closing the pdf file object
pdfFileObj.close()

