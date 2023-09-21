import re

fpi = open('input.txt', 'r')
fpo = open('output.txt', 'w')
line = fpi.readline()

while(line):
	tmp = re.findall(r'\d+', line)
	numbers = list(map(int, tmp))

	if '<<' in line:
		nThread = numbers[2] - numbers[6]
	else:
		nThread = numbers[0] - numbers[2]

	divisible = False
	if nThread >= 1024:
		for blockSize in [1024, 512, 256, 128, 64, 32]:
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
			func_str += 'xor_gpu<<<{}, {}>>>(poly, offset+({}<<{})-{}, '.format(nBlock, nThread, numbers[0], numbers[1], numbers[2])
			func_str += str(numbers[7:])[1:-1]
		else:
			func_str += 'xor_gpu<<<{}, {}>>>(poly, offset-{}, '.format(nBlock, nThread, numbers[0])
			func_str += str(numbers[3:])[1:-1]
		func_str += ');\n'

	fpo.write(func_str)
	line = fpi.readline()
