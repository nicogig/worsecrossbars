import asyncio

async def say():
    print('hello!')

async def main():
    tasks = []
    loop = asyncio.get_event_loop()
    for i in range(4):
        tasks.append(loop.create_task(say()))
    await asyncio.gather(*tasks)

    print('popi')

if __name__ == "__main__":
    asyncio.run(main())