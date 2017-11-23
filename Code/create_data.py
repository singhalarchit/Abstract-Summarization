import glob
import xml.etree.ElementTree as ET

error_files = ['../NLP Dataset\W07-1426-parscit.130908.xml', '../NLP Dataset\W08-2205-parscit.130908.xml', '../NLP Dataset\W11-2102-parscit.130908.xml']

def main():
  abstract_writer = open('../Processed Data/abstracts.txt', 'w')
  title_writer = open('../Processed Data/titles.txt', 'w')
  failure_writer = open('../Processed Data/failure.txt', 'w')
  exception_writer = open('../Processed Data/exception.txt', 'w')
  files = glob.glob('../NLP Dataset/*.xml')
  success, failure, exceptions = 0, 0, 0
  i = 1
  for filename in files:
    print(i)
    i += 1
    try:
      tree = ET.parse(filename)
      root = tree.getroot()
      variant = root.findall('algorithm')[1][0]
      title = variant.findall('title')[0]
      abstract = variant.findall('abstract')#[-1]
      try:
        if filename in error_files:  
          abstract_writer.write(abstract[-1].text.encode('utf-8') + '\n')
        else:
          abstract_writer.write(abstract[0].text.encode('utf-8') + '\n')
        title_writer.write(title.text.encode('utf-8') + '\n')
        success += 1
      except:
        failure += 1
        failure_writer.write(str(filename) + '\n')   
    except:
      exceptions += 1
      exception_writer.write(str(filename) + '\n')
  print(str(success) + " files successful")
  print(str(failure) + " files have no abstract or title")
  print(str(exceptions) + " files exceptions")
  abstract_writer.close()
  title_writer.close()
  failure_writer.close()
  exception_writer.close()

if __name__ == "__main__":
  main()
