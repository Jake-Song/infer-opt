import { expect, test } from '@playwright/test';

test('controls update metrics, roofline, and traffic; keyboard can inspect a point', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('추론 성능');
  const throughput = await page.getByTestId('OUTPUT THROUGHPUT').textContent();
  const traffic = await page.getByTestId('traffic').textContent();
  await page.getByRole('spinbutton', { name: 'Batch size 값' }).fill('32');
  await expect(page.getByTestId('OUTPUT THROUGHPUT')).not.toHaveText(throughput!);
  await expect(page.getByTestId('traffic')).not.toHaveText(traffic!);
  const point = page.getByRole('button', { name: 'Prefill 상세 보기' });
  await point.focus(); await page.keyboard.press('Enter');
  await expect(point).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('slider', { name: 'Batch size', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('spinbutton', { name: 'Batch size 값' })).toHaveValue('33');
  await page.getByRole('combobox', { name: 'Hidden dimension' }).selectOption('2048');
  await page.getByRole('button', { name: 'FP32', exact: true }).click();
  await expect(page.getByRole('button', { name: 'FP32', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(errors).toEqual([]);
});

test('can save, compare, apply every guided experiment, and reset', async ({ page }) => {
  await page.goto('/');
  const initial = await page.getByTestId('TTFT').textContent();
  await page.getByRole('button', { name: '기준 저장' }).click();
  await expect(page.getByRole('status')).toContainText('B1');
  await page.getByRole('spinbutton', { name: 'Prompt length 값' }).fill('4096');
  await expect(page.getByTestId('TTFT')).not.toHaveText(initial!);
  await expect(page.getByRole('status')).toContainText('1,024 prompt');
  for (const name of ['Batch를 키우면?', '문맥이 길어지면?', '연산력 vs 대역폭']) {
    const experiment = page.getByRole('button').filter({ has: page.getByRole('heading', { name, exact: true }) });
    await experiment.click();
    await expect(experiment).toHaveAttribute('aria-pressed', 'true');
  }
  await expect(page.getByRole('spinbutton', { name: '연산 능력', exact: false })).toHaveValue('200');
  await page.getByRole('button', { name: '초기화', exact: true }).click();
  await expect(page.getByRole('status')).toHaveCount(0);
  await expect(page.getByTestId('TTFT')).toHaveText(initial!);
  await expect(page.getByRole('spinbutton', { name: '연산 능력', exact: false })).toHaveValue('100');
});

test('invalid hardware keeps last valid results; one output has no decode', async ({ page }) => {
  await page.goto('/');
  const initial = await page.getByTestId('TTFT').textContent();
  const compute = page.getByRole('spinbutton', { name: '연산 능력', exact: false });
  await compute.fill('0');
  await expect(compute).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByTestId('TTFT')).toHaveText(initial!);
  await compute.fill('100');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByRole('spinbutton', { name: 'Output length 값' }).fill('1');
  await expect(page.getByTestId('AVG. TPOT')).toHaveText('해당 없음');
  await expect(page.getByRole('button', { name: 'Decode', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Decode 상세 보기' })).toHaveCount(0);
  await expect(page.getByTestId('traffic')).toContainText('Prefill');
});

test('numeric controls support partial keyboard edits without changing results to invalid values', async ({ page }) => {
  await page.goto('/');
  const prompt = page.getByRole('spinbutton', { name: 'Prompt length 값' });
  const initial = await page.getByTestId('TTFT').textContent();
  await prompt.fill('');
  await expect(prompt).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByTestId('TTFT')).toHaveText(initial!);
  await prompt.pressSequentially('16');
  await expect(prompt).toHaveValue('16');
  await expect(page.getByRole('slider', { name: 'Prompt length', exact: true })).toHaveValue('16');
  await prompt.fill('99999');
  await prompt.blur();
  await expect(prompt).toHaveValue('16');
});

test('timeline plays, pauses, scrubs, and supports reduced motion', async ({ page }) => {
  await page.goto('/');
  const progress = page.getByRole('slider', { name: '생성 토큰 위치' });
  await page.getByRole('button', { name: '타임라인 재생', exact: true }).click();
  await expect(progress).not.toHaveValue('0');
  await page.getByRole('button', { name: '일시정지', exact: true }).click();
  await progress.fill('20');
  await expect(progress).toHaveValue('20');
  await progress.fill('1');
  await expect(page.locator('.token.first')).toHaveClass(/filled/);
  await progress.fill('20');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.getByRole('button', { name: '다음 토큰', exact: true }).click();
  await expect(progress).toHaveValue('21');
  await page.getByRole('spinbutton', { name: 'Output length 값' }).fill('5');
  await expect(progress).toHaveValue('0');
});

test('renders without overflow and exposes its assumptions', async ({ page }) => {
  await page.goto('/');
  await page.getByText('계산 가정과 한계 알아보기').click();
  await expect(page.getByText('모든 성능 수치는 계산 모델의 하한 시간', { exact: false })).toBeVisible();
  await page.getByText('계산 가정과 한계 알아보기').click();
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
  for (const width of [1440, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.screenshot({ path: 'test-results/narrow.png', fullPage: true });
});

test('extreme hardware comparisons stay inside the roofline plot', async ({ page }) => {
  await page.goto('/');
  const bandwidth = page.getByRole('spinbutton', { name: '메모리 대역폭', exact: false });
  await bandwidth.fill('1');
  await page.getByRole('button', { name: '기준 저장' }).click();
  await bandwidth.fill('100000');
  await page.getByRole('spinbutton', { name: '연산 능력', exact: false }).fill('10000');
  const bounds = await page.locator('.baseline-roof').evaluate(el => {
    const box = (el as SVGGraphicsElement).getBBox();
    return { top: box.y, bottom: box.y + box.height };
  });
  expect(bounds.top).toBeGreaterThanOrEqual(33);
  expect(bounds.bottom).toBeLessThanOrEqual(285);
  await page.getByRole('spinbutton', { name: 'Batch size 값' }).fill('128');
  await page.getByRole('spinbutton', { name: 'Prompt length 값' }).fill('32768');
  await page.getByRole('spinbutton', { name: 'Output length 값' }).fill('512');
  await page.getByRole('combobox', { name: 'Hidden dimension' }).selectOption('4096');
  await bandwidth.fill('1');
  await page.getByRole('spinbutton', { name: '연산 능력', exact: false }).fill('0.1');
  for (const metric of await page.locator('.metric-value').all()) {
    expect(await metric.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    await expect(metric).not.toContainText('NaN');
    await expect(metric).not.toContainText('Infinity');
  }
});
