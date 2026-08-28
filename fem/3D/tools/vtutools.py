#!/usr/bin/env python
import vtk

class VTK_XML_Serial_Unstructured:
  
  """
  USAGE:
  vtk_writer = VTK_XML_Serial_Unstructured()
  vtk_writer.writeVTU("filename.vtu", x, y, z, optional arguments...)
  vtk_writer.writePVD("filename.pvd")
  """
  
  def __init__(self):
    self.fileNames = []
  
  def listPVD(self):
    print ("***********************************")
    print ("PVD list:\n")
    for i in range(len(self.fileNames)):
      print (self.fileNames[i][0])
    print ("***********************************\n")
  
  def addtoPVD(self,fileName, time, vtuExt=".vtu"):
    if vtuExt==".pvtu":
        fileName = fileName + "."
    fileName = fileName + time + vtuExt
    self.fileNames.append([fileName, time])
  
  def readVTU(self, fileName):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fileName)
    reader.Update()
    return reader
    
  def extractVTUinfo(self, reader):
    out = reader.GetOutput()
    
    coo=[]
    for i in range(out.GetNumberOfPoints()):
      p=[0,0,0]
      out.GetPoint(i, p)
      coo.append(p)
      
    cel=[]
    npts = out.GetCell(1).GetNumberOfPoints()
    for i in range(out.GetNumberOfCells()):
      p = [out.GetCell(i).GetPointId(j) for j in range(npts)]
      cel.append(p)
   
    da=[]
    la=[]
    for i in range(out.GetPointData().GetNumberOfArrays()):
      ar = out.GetPointData().GetArray(i)
      npts = ar.GetNumberOfTuples()
      la.append(ar.GetName())
      p = [str(ar.GetTuple(j)).replace(",","\t").replace("(","").replace(")","") for j in range(npts) ]
      da.append(p)
    
#    print "Import completed"
    return coo, cel, da, la
  
  def appendVTU(self, listf):
    appendF = vtk.vtkAppendFilter()
    for l in listf:
      appendF.AddInputConnection(l.GetOutputPort())
      appendF.Update()
    return appendF
    
  def contourVTU(self, reader, scalar, value):
    readerS = vtk.vtkAssignAttribute()
    readerS.SetInputConnection(reader.GetOutputPort())
    readerS.Assign(scalar, "SCALARS", "POINT_DATA")
    readerS.Update()

    conVtk = vtk.vtkContourFilter()
    conVtk.SetInputData(readerS.GetOutput())
    conVtk.SetValue(0,value)
    conVtk.ComputeNormalsOff()
    conVtk.Update()

    if(conVtk.GetOutput().GetNumberOfPoints() * conVtk.GetOutput().GetNumberOfCells() == 0):
        raise("Invalid contour")
    
#    print "Contour completed"
    return conVtk
 
  def clipVTU(self, reader, origin=None, normal=(0,0,1)):
    plane = vtk.vtkPlane()
    if origin == None:
        origin = reader.GetOutput().GetCenter()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    clipVtk = vtk.vtkClipDataSet()
    clipVtk.SetInputData(reader.GetOutput())
    clipVtk.SetClipFunction(plane)
    clipVtk.InsideOutOff()
    clipVtk.Update()
    
    return clipVtk
 
 
  def writeVTU(self, fileName, time, coordinates, elements, data=None, labels=None):
    
    nco = len(coordinates)
    nel = len(elements)
    
    dow = len(coordinates[1])
    off = len(elements[1])
    if dow==2:
      for i in range(len(coordinates)):
          coordinates[i].append("0.")
      if off==3:
          type_el="5"
      else:
          print ("Invalid type element")
          exit()

    else:
      if off==3:
          type_el="5"
      elif off==2:
          type_el="4"
      elif off==4:
          type_el="10"
      else:
          print ("Invalid type element")
          exit()

    # Start write
    
    fileName = fileName + time + ".vtu"
    outFile = open(fileName, 'w')

    outFile.write('<?xml version="1.0"?>\n'
                  '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
                  '  <UnstructuredGrid>\n'
                  '    <Piece NumberOfPoints="%i" NumberOfCells="%i">\n' % (nco,nel) )
    # Points
    outFile.write('      <Points>\n'
		  '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
    for co in coordinates:
      outFile.write('\t'.join(str(x) for x in co) + '\n')

    outFile.write('        </DataArray>\n'
		  '      </Points>\n')
    #Cells
    outFile.write('      <Cells>\n'
		  '        <DataArray type="Int32" Name="offsets">\n')
    
    for i in range(nel):
      outFile.write('%s\n' % str((i+1)*off))

    outFile.write('        </DataArray>\n'
		  '        <DataArray type="UInt8" Name="types">\n')
    for i in range(nel):
      outFile.write('%s\n' % type_el)

    outFile.write('        </DataArray>\n')
    outFile.write('        <DataArray type="Int32" Name="connectivity">\n');
    for el in elements:
      outFile.write('\t'.join(str(x) for x in el) + '\n')
    outFile.write('        </DataArray>\n');
    outFile.write('      </Cells>\n');

    #PointData
    if data is not None:
      outFile.write('      <PointData>\n');
      for i in range(len(data)):
          outFile.write('        <DataArray type="Float32" Name="%s" format="ascii">\n' % labels[i] )
          outFile.write('\n'.join(str(x) for x in data[i]) + '\n')
          outFile.write('        </DataArray>\n')
      outFile.write('      </PointData>\n')


    outFile.write('    </Piece>\n'
		  '  </UnstructuredGrid>\n'
		  '</VTKFile>\n')

    outFile.close()
    
    self.fileNames.append([fileName,time])
    
    print ("%s written!" % fileName)



  def writeXMLvtu(self, fileName, time, reader): #TODO: it doesn't work properly
    
    unstructuredGrid = vtk.vtkUnstructuredGrid()
    unstructuredGrid.DeepCopy( reader.GetOutput() )
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName( fileName + time + '.vtu' )
    writer.SetInputData(unstructuredGrid)
    writer.SetDataModeToAscii()    #BUG http://www.paraview.org/Bug/view.php?id=13382
    writer.SetCompressorTypeToNone()
    writer.Write()

    self.fileNames.append([fileName,time])

    print ("%s written!" % fileName)



  def writeXYZ(self, fileName, time, coordinates, data=None, labels=None):

    xyzname = fileName+time+".dat"
    xyz = []

    foutXYZ = open(xyzname,'w')
    foutXYZ.write("# x \t y \t z \t")
    foutXYZ.write('\t'.join(str(x) for x in labels) + "\n")
    for ipt in range(0, len(coordinates)):
      line = '\t'.join(str(x) for x in coordinates[ipt]) + '\t'
      line = line + '\t'.join(str(data[j][ipt]) for j in range(0, len(data)))
      xyz.append(line)
    xyz.sort(key=lambda x: float(x.split('\t')[0]))
    foutXYZ.write('\n'.join(str(l) for l in xyz))



  def writePVD(self, fileName):
    outFile = open(fileName, 'w')
    import xml.dom.minidom

    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd.appendChild(pvd_root)

    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)

    for i in range(len(self.fileNames)):
      dataSet = pvd.createElementNS("VTK", "DataSet")
      dataSet.setAttribute("timestep", str(self.fileNames[i][1]))
      dataSet.setAttribute("part", "0")
      dataSet.setAttribute("file", str(self.fileNames[i][0]))
      collection.appendChild(dataSet)

    outFile = open(fileName, 'w')
    pvd.writexml(outFile, newl='\n', indent="  ", addindent="  ")
    outFile.close()
    
    print ("%s written!\n" % fileName  )
