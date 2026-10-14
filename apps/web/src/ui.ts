/** Clases Tailwind de patrones repetidos (equivalentes 1:1 del CSS anterior). */

const surface = 'bg-panel border border-panel-line rounded-[13px] shadow-[0_16px_35px_#00000018] min-w-0'
export const panel = `${surface} p-[23px]`
/** Panel sin padding, para variantes que fijan el suyo. */
export const panelBare = surface

export const panelHead = 'flex items-center justify-between mb-[22px] upto1100:flex-wrap upto1100:gap-[10px]'
export const panelTitle = 'font-display text-[16px] font-semibold m-0'
export const panelTitleSub = 'text-[#c3cdd9]'
export const step = 'text-[9px] tracking-[1.5px] text-[#8497b6] bg-control border border-[#2c3b50] py-[6px] px-[9px] rounded-[5px]'

export const desc = 'text-[12px] text-desc leading-[1.6]'
export const hint = 'text-[11px] text-desc leading-[1.6] mt-[15px]'
export const strong = 'text-ink'
export const errorText = 'text-[#fa9da9] text-[12px]'
export const errorBanner = 'bg-[#482530] border border-[#8c4054] text-[#ffacb1] p-[13px] rounded-[8px] mb-[16px] text-[12px] flex justify-between gap-[8px]'
export const errorClose = 'bg-transparent text-inherit border-0'

export const primaryBtn = 'w-full flex items-center justify-center gap-[12px] bg-transparent bg-[linear-gradient(95deg,#6465f0,#8278ee)] border-0 rounded-[8px] py-[14px] px-[18px] text-[12px] font-bold text-white shadow-[0_5px_22px_#635af039] disabled:opacity-45 disabled:cursor-not-allowed'
export const ghostBtn = 'bg-control border border-control-line text-control-text rounded-[6px] py-[8px] px-[12px] text-[12px] font-semibold cursor-pointer hover:border-accent'
export const linkBtn = 'bg-transparent border-0 text-[#9b9cff] text-[12px] font-semibold cursor-pointer py-[2px] px-[4px] ml-[8px] disabled:text-[#56637a] disabled:cursor-not-allowed'

export const kbd = 'font-[monospace] text-[10px] font-semibold border border-b-2 rounded-[4px] py-[1px] px-[5px] bg-control border-control-line text-control-text'
export const kbdOnPrimary = 'font-[monospace] text-[10px] font-semibold border border-b-2 rounded-[4px] py-[1px] px-[5px] bg-[#ffffff22] border-[#ffffff44] text-white'

export const liveDot = 'bg-live rounded-[50%] w-[7px] h-[7px] inline-block shadow-[0_0_10px_#3ad2a77d]'

export const metrics = 'flex gap-[12px] my-[22px]'
export const metric = 'flex-1 border border-[#2b3949] bg-field rounded-[8px] py-[12px] px-[11px] min-w-0'
export const metricLabel = 'block text-[#7086a0] text-[9px] tracking-[1px] mb-[10px]'
export const metricValue = 'text-[14px] whitespace-nowrap overflow-hidden text-ellipsis block'

export const empty = 'flex min-h-[414px] flex-col justify-center items-center text-center text-[#899cb5]'
export const emptyIcon = 'w-[75px] h-[75px] grid place-items-center text-[32px] text-[#7084a9] rounded-[50%] bg-[#1d2839] border border-[#2a3a4e]'
export const emptyTitle = 'font-display text-[#d7e1ef] text-[16px] mt-[21px] mx-0 mb-[6px]'
export const emptyText = 'text-[12px] text-[#8294ad]'

export const TONES = {
  good: 'bg-ok-bg text-ok', warn: 'bg-warn-bg text-warn', bad: 'bg-bad-bg text-bad',
} as const
