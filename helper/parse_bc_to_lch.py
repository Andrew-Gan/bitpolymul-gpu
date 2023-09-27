import re

fpi = open('input.txt', 'r')
fpo = open('output.txt', 'w')

# outer loop
line = fpi.readline()
tmp = re.findall(r'-?\d+', line)
numbers = list(map(int, tmp))
outIdxStart = numbers[1]
outIdxStep = numbers[4] + numbers[5]
outIterCountStr = '((1<<logn)-(1<<{}))/(1<<{})'.format(outIdxStart, outIdxStep)
fpo.write('blockY = ' + outIterCountStr + ';\n')

# inner loops
line = fpi.readline()
while(line):
	tmp = re.findall(r'-?\d+', line)
	numbers = list(map(int, tmp))
	if len(numbers) == 0:
		break

	if '<<' in line:
		nThread = numbers[3] - numbers[6]
	else:
		nThread = numbers[1] - numbers[2]

	divisible = False
	if nThread >= 1024:
		for blockSize in [1024, 512, 256, 128]:
			if nThread % blockSize == 0:
				nBlock = nThread // blockSize
				nThread = blockSize
				divisible = True
				break
	else:
		divisible = True
		nBlock = 1
	
	func_str = ''
	if not divisible:
		func_str = '{}\n'.format(nThread)
		func_str += line
	else:
		if '<<' in line:
			func_str += 'xor_gpu<<<dim3({}, blockY), {}>>>(poly, (1<<{}){}, 1<<{}, 1<<{}, '.format(nBlock, nThread, numbers[5], numbers[6], outIdxStart, outIdxStep)
			func_str += str(numbers[7:])[1:-1]
		else:
			func_str += 'xor_gpu<<<dim3({}, blockY), {}>>>(poly, {}, 1<<{}, 1<<{}, '.format(nBlock, nThread, numbers[2], outIdxStart, outIdxStep)
			func_str += str(numbers[3:])[1:-1]
		func_str += ');\n'

	fpo.write(func_str)
	line = fpi.readline()
