import EigenFace as ef


imgs = ef.readAllImages('att_faces/s1')

averageFace = ef.getAverageFace(imgs)

_, orth_eigenVectors = ef.getEngineFaces(imgs)

print(orth_eigenVectors.shape)