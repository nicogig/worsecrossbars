import asyncio
from asyncio.tasks import sleep
from os import wait

async def say():
    print('hello!')
    await sleep(2)
    print('world')

async def main():
    tasks = []
    loop = asyncio.get_event_loop()
    for i in range(4):
        tasks.append(loop.create_task(say()))
    await asyncio.gather(*tasks)

    print('popi')

if __name__ == "__main__":
    asyncio.run(main())